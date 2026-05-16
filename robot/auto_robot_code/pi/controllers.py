"""
controllers.py
==============
Cac bo dieu khien va ham toan hoc co ban:
  - wrap_angle_rad, signed_deadband
  - Kalman1D, PID, SlewRateLimiter
  - AxisPD, AlignmentAutoTuner
"""
import math
import time

import numpy as np

from config import (
    KALMAN_Q, KALMAN_R, KALMAN_VEL_DECAY, CONTROL_RATE,
    ADAPT_GAIN_MIN, ADAPT_GAIN_MAX, ADAPT_OVERSHOOT_ERR, ADAPT_OVERSHOOT_RATE,
    ADAPT_DROP, ADAPT_RECOVER_PER_S, ALIGN_STRAFE_ENTER, ALIGN_RATE_HOLD,
    ALIGN_HOLD_ENTER,
)


def signed_deadband(value, deadband):
    if abs(value) <= deadband:
        return 0.0
    return math.copysign(abs(value) - deadband, value)


def wrap_angle_rad(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Kalman1D:
    def __init__(self, q=KALMAN_Q, r=KALMAN_R):
        self.x = 0.0
        self.dx = 0.0
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.q = q
        self.r = r
        self.initialized = False

    def reset(self):
        self.x = 0.0
        self.dx = 0.0
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.initialized = False

    def init_with(self, value):
        self.x = value
        self.dx = 0.0
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return self.x

        self.dx *= KALMAN_VEL_DECAY
        self.x += dt * self.dx

        dt2 = dt * dt
        q = self.q
        self.P[0][0] += dt * (self.P[0][1] + self.P[1][0]) + dt2 * self.P[1][1] + q * dt2 * dt2 / 4.0
        self.P[0][1] += dt * self.P[1][1] + q * dt2 * dt / 2.0
        self.P[1][0] += dt * self.P[1][1] + q * dt2 * dt / 2.0
        self.P[1][1] += q * dt2
        return self.x

    def update(self, measurement, dt):
        if not self.initialized:
            self.init_with(measurement)
            return self.x

        self.predict(dt)

        innovation = measurement - self.x
        s_val = self.P[0][0] + self.r
        k0 = self.P[0][0] / s_val
        k1 = self.P[1][0] / s_val

        self.x += k0 * innovation
        self.dx += k1 * innovation

        p00, p01 = self.P[0][0], self.P[0][1]
        p10, p11 = self.P[1][0], self.P[1][1]
        self.P[0][0] = (1 - k0) * p00
        self.P[0][1] = (1 - k0) * p01
        self.P[1][0] = p10 - k1 * p00
        self.P[1][1] = p11 - k1 * p01
        return self.x


class PID:
    def __init__(self, kp, ki, kd, out_min, out_max, derivative_tau=0.08):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.derivative_tau = derivative_tau
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.d_state = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.d_state = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def compute(self, error, dt=None, feedforward=0.0):
        now = time.monotonic()
        if dt is None:
            if self.prev_time is None:
                dt = 1.0 / CONTROL_RATE
            else:
                dt = now - self.prev_time
            if dt <= 0:
                dt = 0.001
        self.prev_time = now

        self.last_p = self.kp * error
        raw_d = (error - self.prev_error) / max(dt, 1e-3)
        alpha = dt / (self.derivative_tau + dt)
        self.d_state += alpha * (raw_d - self.d_state)
        self.last_d = self.kd * self.d_state

        if abs(error) < 0.02 and abs(self.prev_error) < 0.02:
            self.integral *= 0.85

        tentative_integral = self.integral + error * dt
        tentative_i = self.ki * tentative_integral
        u_pre = self.last_p + tentative_i + self.last_d + feedforward
        u_sat = float(np.clip(u_pre, self.out_min, self.out_max))
        allow_integral = abs(u_pre - u_sat) < 1e-6 or np.sign(error) != np.sign(u_pre - u_sat)

        if allow_integral:
            self.integral = tentative_integral

        max_integral = abs(self.out_max / (self.ki + 1e-9))
        self.integral = float(np.clip(self.integral, -max_integral, max_integral))
        self.last_i = self.ki * self.integral
        self.prev_error = error

        output = self.last_p + self.last_i + self.last_d + feedforward
        return float(np.clip(output, self.out_min, self.out_max))


class SlewRateLimiter:
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.value = 0.0
        self.initialized = True

    def reset(self, value=0.0):
        self.value = float(value)
        self.initialized = True

    def update(self, target, dt):
        if not self.initialized:
            self.value = float(target)
            self.initialized = True
            return self.value

        max_delta = self.rate_limit * max(dt, 1e-3)
        delta = float(np.clip(target - self.value, -max_delta, max_delta))
        self.value += delta
        return self.value


class AxisPD:
    def __init__(self, kp, kd, out_min, out_max):
        self.kp = kp
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.last_p = 0.0
        self.last_d = 0.0

    def reset(self):
        self.last_p = 0.0
        self.last_d = 0.0

    def compute(self, error, error_rate, scale=1.0):
        self.last_p = -self.kp * error
        self.last_d = -self.kd * error_rate
        output = scale * (self.last_p + self.last_d)
        return float(np.clip(output, self.out_min, self.out_max))


class AlignmentAutoTuner:
    def __init__(self):
        self.gain_scale = 1.0
        self.last_sign = 0
        self.overshoots = 0
        self.stable_time = 0.0

    def reset(self):
        self.gain_scale = 1.0
        self.last_sign = 0
        self.overshoots = 0
        self.stable_time = 0.0

    def update(self, err_x, err_rate, dt):
        magnitude = abs(err_x)
        sign = 0
        if magnitude >= ADAPT_OVERSHOOT_ERR:
            sign = 1 if err_x > 0.0 else -1

        crossed = (
            self.last_sign != 0
            and sign != 0
            and sign != self.last_sign
            and abs(err_rate) > ADAPT_OVERSHOOT_RATE
        )

        if crossed:
            self.overshoots += 1
            self.gain_scale = max(ADAPT_GAIN_MIN, self.gain_scale * (1.0 - ADAPT_DROP))
            self.stable_time = 0.0
        elif magnitude < ALIGN_STRAFE_ENTER and abs(err_rate) < ALIGN_RATE_HOLD:
            self.stable_time += dt
            if self.stable_time > 0.35:
                self.gain_scale = min(ADAPT_GAIN_MAX, self.gain_scale + ADAPT_RECOVER_PER_S * dt)
        else:
            self.stable_time = 0.0

        if sign != 0:
            self.last_sign = sign

        return self.gain_scale, crossed
