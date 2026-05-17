# Lyapunov Controller for Front-Differential Tricycle Geometry

This note designs a Lyapunov-style kinematic controller for a vehicle with:

- two front driven wheels separated by track width \(a\);
- one centered rear passive caster wheel;
- distance from front axle to rear caster wheel \(b\);
- rear caster steering angle limit \(|\gamma| \le 100^\circ\);
- center of gravity on the centerline, assumed halfway between front axle and rear wheel unless otherwise measured.

The model is kinematic. It is not a tire-force or actuator-dynamics model.

## 1. Geometry and Control Point

Use the midpoint of the front axle as the control point \(P\):

$$
p = [x, y]^T,\qquad \theta = \text{body yaw}.
$$

The front wheels are differentially driven. If the left and right front wheel ground speeds are \(v_L\) and \(v_R\), define:

$$
v = \frac{v_R + v_L}{2}
$$

$$
\omega = \frac{v_R - v_L}{a}
$$

Then the kinematics of the front-axle midpoint are:

$$
\dot{x} = v\cos\theta
$$

$$
\dot{y} = v\sin\theta
$$

$$
\dot{\theta} = \omega
$$

So the controller can first generate the virtual inputs:

$$
u = [v,\omega]^T
$$

and then map them to front wheel speeds:

$$
v_R = v + \frac{a}{2}\omega
$$

$$
v_L = v - \frac{a}{2}\omega
$$

Here \(a\) affects the wheel-speed mapping, not the unicycle-like control law itself.

## 2. Rear Passive Caster Constraint

The rear caster is located at:

$$
r_R = [-b, 0]^T
$$

in the body frame, measured from the front-axle midpoint.

The velocity of the rear contact point in the body frame is:

$$
v_{rear}^{body}
=
\begin{bmatrix}
v \\
-b\omega
\end{bmatrix}
$$

The passive rear wheel aligns with this velocity. Therefore the required rear caster angle is:

$$
\gamma = \operatorname{atan2}(-b\omega, v)
$$

The mechanical limit is:

$$
|\gamma| \le \gamma_{max}
$$

with:

$$
\gamma_{max} = 100^\circ = 1.745\ \mathrm{rad}
$$

Important practical consequence:

- If \(v \ge 0\), then \(|\gamma| \le 90^\circ\) for finite \(\omega\), so the \(100^\circ\) limit is normally satisfied.
- Backward motion can violate the caster limit because the rear wheel may need to rotate close to \(180^\circ\).
- Therefore, a safe implementation should prefer forward motion and turn-in-place rather than large reverse commands.

## 3. Center of Gravity

Let the center of gravity \(C\) lie on the centerline at distance \(c\) behind the front axle:

$$
c = \frac{b}{2}
$$

if "centered" means halfway between front axle and rear caster.

The relation between front-axle midpoint \(P\) and center of gravity \(C\) is:

$$
p_C = p_P - c
\begin{bmatrix}
\cos\theta \\
\sin\theta
\end{bmatrix}
$$

Equivalently:

$$
p_P = p_C + c
\begin{bmatrix}
\cos\theta \\
\sin\theta
\end{bmatrix}
$$

The Lyapunov controller below is simplest when tracking \(P\). If the desired trajectory is given for the center of gravity, convert it to a desired front-axle trajectory using the equation above.

## 4. Tracking Error

Assume a desired front-axle reference trajectory:

$$
x_d(t),\quad y_d(t),\quad \theta_d(t)
$$

with desired virtual inputs:

$$
v_d(t),\quad \omega_d(t)
$$

satisfying:

$$
\dot{x}_d = v_d\cos\theta_d
$$

$$
\dot{y}_d = v_d\sin\theta_d
$$

$$
\dot{\theta}_d = \omega_d
$$

Define body-frame tracking error:

$$
e_x = \cos\theta(x_d-x) + \sin\theta(y_d-y)
$$

$$
e_y = -\sin\theta(x_d-x) + \cos\theta(y_d-y)
$$

$$
e_\theta = \operatorname{wrap}(\theta_d-\theta)
$$

Interpretation:

- \(e_x\): longitudinal error in the vehicle frame;
- \(e_y\): lateral error in the vehicle frame;
- \(e_\theta\): heading error.

## 5. Error Dynamics

For the front-axle unicycle-like kinematics, the error dynamics are:

$$
\dot{e}_x = \omega e_y - v + v_d\cos e_\theta
$$

$$
\dot{e}_y = -\omega e_x + v_d\sin e_\theta
$$

$$
\dot{e}_\theta = \omega_d - \omega
$$

## 6. Lyapunov Function

Use:

$$
V
=
\frac{1}{2}(e_x^2+e_y^2)
+ \frac{1}{k_y}(1-\cos e_\theta)
$$

with positive gains:

$$
k_x>0,\quad k_y>0,\quad k_\theta>0
$$

This function is positive definite around:

$$
e_x=0,\quad e_y=0,\quad e_\theta=0.
$$

## 7. Control Law

Choose:

$$
v
=
v_d\cos e_\theta + k_x e_x
$$

$$
\omega
=
\omega_d + k_y v_d e_y + k_\theta \sin e_\theta
$$

Then:

$$
\dot{V}
=
-k_x e_x^2
-\frac{k_\theta}{k_y}\sin^2 e_\theta
\le 0
$$

Thus \(e_x\) and \(e_\theta\) are directly damped. With a persistently moving reference \(v_d\neq 0\), the lateral error \(e_y\) is also driven to zero through the coupling term \(k_y v_d e_y\).

This is a trajectory-tracking controller. For a completely static target pose, a separate parking/alignment supervisor is recommended because smooth global stabilization of a unicycle-like system has known limitations.

## 8. Wheel-Speed Commands

After computing \(v\) and \(\omega\), command the front wheel speeds:

$$
v_R = v + \frac{a}{2}\omega
$$

$$
v_L = v - \frac{a}{2}\omega
$$

If wheel angular speeds are required and the front wheel radius is \(R_f\):

$$
\Omega_R = \frac{v_R}{R_f}
$$

$$
\Omega_L = \frac{v_L}{R_f}
$$

## 9. Saturation and Caster Safety

The Lyapunov proof assumes no saturation. The real implementation should apply safety limits after computing the raw command.

Recommended order:

1. Compute raw \(v,\omega\) from the Lyapunov law.
2. Prefer forward motion:

$$
v \leftarrow \max(v,0)
$$

if the rear caster cannot safely reverse.

3. Check rear caster angle:

$$
\gamma = \operatorname{atan2}(-b\omega,v)
$$

4. If:

$$
|\gamma| > \gamma_{max}
$$

reduce \(|\omega|\), increase \(v\), or switch to a turn-in-place maneuver depending on the mechanical design.

5. Convert to wheel speeds \(v_L,v_R\).
6. Saturate wheel speeds.
7. If wheel-speed saturation changes the ratio between \(v\) and \(\omega\), recompute the actual \(v,\omega\) used for monitoring.

Because \(\gamma_{max}=100^\circ\), the rear caster limit is usually satisfied for \(v\ge0\). The risky case is reverse motion.

## 10. Practical Gain Choice

A reasonable initial tuning structure is:

$$
k_x = 1.0 \sim 3.0
$$

$$
k_y = 2.0 \sim 6.0
$$

$$
k_\theta = 1.0 \sim 4.0
$$

Guidelines:

- Increase \(k_x\) to correct forward/backward error faster.
- Increase \(k_y\) to correct lateral error more aggressively.
- Increase \(k_\theta\) to align heading faster.
- Do not tune gains without considering wheel-speed saturation.
- If commands oscillate, reduce \(k_y\) and \(k_\theta\), or add rate limits.

## 11. Final Controller Summary

Inputs:

$$
x,y,\theta,\quad x_d,y_d,\theta_d,\quad v_d,\omega_d
$$

Geometry:

$$
a,\quad b,\quad c=b/2,\quad \gamma_{max}=100^\circ
$$

Error:

$$
e_x = \cos\theta(x_d-x) + \sin\theta(y_d-y)
$$

$$
e_y = -\sin\theta(x_d-x) + \cos\theta(y_d-y)
$$

$$
e_\theta = \operatorname{wrap}(\theta_d-\theta)
$$

Virtual control:

$$
v
=
v_d\cos e_\theta + k_x e_x
$$

$$
\omega
=
\omega_d + k_y v_d e_y + k_\theta \sin e_\theta
$$

Front wheel commands:

$$
v_R = v + \frac{a}{2}\omega
$$

$$
v_L = v - \frac{a}{2}\omega
$$

Rear caster monitoring:

$$
\gamma = \operatorname{atan2}(-b\omega,v)
$$

Constraint:

$$
|\gamma| \le 100^\circ
$$

