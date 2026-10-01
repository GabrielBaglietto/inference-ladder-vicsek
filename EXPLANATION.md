# Running a flock backwards: a detailed explanation

*The mathematical, physical and statistical ideas behind the [article](README.md), developed step by step.*

by [Gabriel Baglietto](https://github.com/GabrielBaglietto)

The [article](README.md) tells the story for a reader who already works with models like this one, and keeps
its mathematics folded. The [notebooks](notebooks/) hold every computation. This document sits between the
two: it takes each idea and develops it slowly, with the mathematics written out, for a reader who knows how
to simulate a model forward but has not done inference on it.

The central problem is an **inverse problem for a stochastic dynamical system**:

> If we observe only the positions of particles in a flock, can we infer the microscopic rules that generated
> their motion?

We answer it with the Vicsek model and a sequence of increasingly powerful inference tools: observables,
sensitivity analysis, transient timescales, synthetic likelihood, Fisher information, parametric bootstrap,
and Bayesian posterior inference.

![Forty particles at three noise levels](figures/three_noise_levels.gif)

## Contents

**Part I — The problem and the model**
1. [Forward and inverse problems](#1-forward-and-inverse-problems)
2. [The physical model: the Vicsek model](#2-the-physical-model-the-vicsek-model)
3. [Step 1 of the dynamics: find the neighbours](#3-step-1-of-the-dynamics-find-the-neighbours)
4. [Step 2: the neighbours' average direction](#4-step-2-the-neighbours-average-direction)
5. [Step 3: add angular noise](#5-step-3-add-angular-noise)
6. [Step 4: move the particle](#6-step-4-move-the-particle)
7. [Periodic boundary conditions](#7-periodic-boundary-conditions)
8. [The order parameter](#8-the-order-parameter)

**Part II — What we observe, and what we compute from it**

9. [The key observational problem](#9-the-key-observational-problem)
10. [Net displacement and apparent speed](#10-net-displacement-and-apparent-speed)
11. [Turning angles from three frames](#11-turning-angles-from-three-frames)
12. [Why hidden headings are not fatal](#12-why-hidden-headings-are-not-fatal)
13. [Why use summary statistics?](#13-why-use-summary-statistics)
14. [Group I: individual motion](#14-group-i-individual-motion)
15. [Group II: collective alignment](#15-group-ii-collective-alignment)
16. [Group III: flock geometry](#16-group-iii-flock-geometry)
17. [Why averaging over time does not eliminate the noise](#17-why-averaging-over-time-does-not-eliminate-the-noise)

**Part III — Which observable sees which rule**

18. [A response must be judged against the noise](#18-a-response-must-be-judged-against-the-noise)
19. [The sensitivity z](#19-the-sensitivity-z)
20. [Why the sign matters](#20-why-the-sign-matters)
21. [Interpreting z = 2](#21-interpreting-z--2)
22. [What the sensitivity map shows](#22-what-the-sensitivity-map-shows)

**Part IV — A rate from the transient**

23. [A characteristic time of ordering](#23-a-characteristic-time-of-ordering)
24. [The half-time](#24-the-half-time)
25. [Why the speed influences the geometric relaxation](#25-why-the-speed-influences-the-geometric-relaxation)
26. [What the half-time gives, and what the next step adds](#26-what-the-half-time-gives-and-what-the-next-step-adds)

**Part V — The synthetic likelihood**

27. [Ordinary likelihood](#27-ordinary-likelihood)
28. [Why the exact likelihood is unavailable](#28-why-the-exact-likelihood-is-unavailable)
29. [Synthetic likelihood](#29-synthetic-likelihood)
30. [The multivariate Gaussian likelihood](#30-the-multivariate-gaussian-likelihood)
31. [First term: a distance that knows the noise](#31-first-term-a-distance-that-knows-the-noise)
32. [Correlations matter too](#32-correlations-matter-too)
33. [Second term: the volume of the cloud](#33-second-term-the-volume-of-the-cloud)
34. [A geometric interpretation](#34-a-geometric-interpretation)
35. [The fit](#35-the-fit)
36. [Parameter-space geometry: valleys and ridges](#36-parameter-space-geometry-valleys-and-ridges)
37. [The valley between noise and speed](#37-the-valley-between-noise-and-speed)
38. [Why a narrow curved valley makes the search delicate](#38-why-a-narrow-curved-valley-makes-the-search-delicate)

**Part VI — Identifiability and the Fisher information**

39. [Identifiability](#39-identifiability)
40. [A simple identifiability example](#40-a-simple-identifiability-example)
41. [The degeneracy between noise and radius](#41-the-degeneracy-between-noise-and-radius)
42. [Why the turning angle resolves the degeneracy](#42-why-the-turning-angle-resolves-the-degeneracy)
43. [Fisher information](#43-fisher-information)
44. [The Fisher information we use](#44-the-fisher-information-we-use)
45. [Deriving the Fisher information, and what it leaves out](#45-deriving-the-fisher-information-and-what-it-leaves-out)
46. [Why the inverse of I gives the uncertainty of the estimate](#46-why-the-inverse-of-i-gives-the-uncertainty-of-the-estimate)
47. [Why log-parameter errors are useful](#47-why-log-parameter-errors-are-useful)
48. [Error correlations](#48-error-correlations)
49. [Leave-one-summary-out analysis](#49-leave-one-summary-out-analysis)
50. [Removing the apparent speed](#50-removing-the-apparent-speed)
51. [Removing the turning angle](#51-removing-the-turning-angle)
52. [Removing the polarization](#52-removing-the-polarization)
53. [Information grows as signal squared over noise squared](#53-information-grows-as-signal-squared-over-noise-squared)

**Part VII — From curvature to the full uncertainty**

54. [What the curvature assumes](#54-what-the-curvature-assumes)
55. [Parametric bootstrap](#55-parametric-bootstrap)
56. [What the bootstrap adds to the curvature](#56-what-the-bootstrap-adds-to-the-curvature)
57. [Two different scales of likelihood variation](#57-two-different-scales-of-likelihood-variation)
58. [Why maximising inherits the roughness](#58-why-maximising-inherits-the-roughness)
59. [Bayesian inference](#59-bayesian-inference)
60. [Why a posterior is particularly useful for ridges](#60-why-a-posterior-is-particularly-useful-for-ridges)
61. [MCMC](#61-mcmc)
62. [A likelihood that is itself simulated](#62-a-likelihood-that-is-itself-simulated)
63. [The pseudo-marginal principle](#63-the-pseudo-marginal-principle)
64. [Which posterior is "exact"?](#64-which-posterior-is-exact)
65. [Why the plain synthetic likelihood is biased](#65-why-the-plain-synthetic-likelihood-is-biased)
66. [Posterior results](#66-posterior-results)
67. [The posterior correlation between noise and speed](#67-the-posterior-correlation-between-noise-and-speed)

**Part VIII — What carries over to other problems**

68. [Three things that are easy to confuse](#68-three-things-that-are-easy-to-confuse)
69. [Better observables, not more of them](#69-better-observables-not-more-of-them)
70. [Precise can beat varied: the dense box](#70-precise-can-beat-varied-the-dense-box)
71. [The camera is part of the model](#71-the-camera-is-part-of-the-model)
72. [A number is not a result](#72-a-number-is-not-a-result)
73. [What it costs](#73-what-it-costs)
74. [The ladder at a glance](#74-the-ladder-at-a-glance)
75. [What the study shows, and what it does not](#75-what-the-study-shows-and-what-it-does-not)
76. [The whole chain](#76-the-whole-chain)
77. [The essential equations on one page](#77-the-essential-equations-on-one-page)
78. [Six ideas to keep](#78-six-ideas-to-keep)

---

# Part I — The problem and the model

## 1. Forward and inverse problems

A **forward problem** starts from known physical rules and predicts observations.

Suppose the rules of a flock are specified by three numbers,

$$\theta = (\eta, r, v_0),$$

where

- η is the amplitude of the angular noise,
- r is the interaction radius,
- v₀ is the particle speed.

θ is the **vector of parameters**. (The heading of particle i is an angle too; we call it αᵢ, so that the two
are never confused.)

If these numbers are known, we can simulate the system:

$$(\eta, r, v_0) \longrightarrow \text{trajectories}.$$

That is the easy direction. We study the opposite arrow:

$$\text{observed trajectories} \longrightarrow (\eta, r, v_0).$$

This is an **inverse problem**.

The distinction is fundamental. Being able to simulate a physical system does not imply that its parameters
can be reconstructed from observations. There are at least three difficulties.

### 1.1 Randomness

Even if two experiments have exactly the same parameters, $\theta_1 = \theta_2$, they will not generate
identical trajectories, because the Vicsek model contains stochastic angular noise. Thus

$$\theta \not\longrightarrow \text{one unique trajectory}.$$

Instead,

$$\theta \longrightarrow \text{a probability distribution over trajectories}.$$

So parameter inference becomes statistical.

### 1.2 Partial observation

The microscopic Vicsek rule acts on headings, but our simulated camera records only positions. The hidden
variables are the headings $\alpha_i(t)$; the measured variables are the positions $\mathbf x_i(t)$.
Therefore we are trying to infer rules that act on an unobserved quantity.

### 1.3 Different parameters may imitate one another

Imagine that increasing one parameter moves an observable upward, while increasing another moves it downward.
Then combinations such as

$$(\theta_1 + \Delta\theta_1,\ \theta_2 + \Delta\theta_2)$$

might leave the observable almost unchanged. The data may therefore determine a *combination* of parameters
much better than either parameter separately. This is the issue of **identifiability**, one of the major
themes of the study.

## 2. The physical model: the Vicsek model

We use the model of [Vicsek et al. (1995)](README.md#references), with N = 40 particles. Particle i has

- a position $\mathbf x_i(t) = (x_i(t), y_i(t))$,
- and a heading $\alpha_i(t)$.

The system lives in a square periodic box of side L = 10. The parameter values that generate our synthetic
observations are

$$\eta^* = 2, \qquad r^* = 1, \qquad v_0^* = 0.03,$$

and they are deliberately hidden from the inference machinery. The interaction radius r is the **unit of
length**: all distances are measured in units of r, and v₀ is a distance per step. With 40 particles in a box
of side 10 there are about 1.3 neighbours per particle within r, so groups form, split and merge; in a much
denser box the whole flock would be a single group. One "experiment" is 5000 steps, observed after the flock
has forgotten its random start.

One step of the dynamics has the parts described in the next four sections.

## 3. Step 1 of the dynamics: find the neighbours

Particle i interacts with the particles j satisfying

$$|\mathbf x_j - \mathbf x_i| \le r,$$

itself included, with distances measured across the periodic boundaries. We call this the neighbour set
$\mathcal N_i(t)$.

The important point is that the interaction is **local**. A particle does not ask:

> Which way is the entire flock moving?

It asks:

> Which way are the particles near me moving?

The radius r therefore controls how far information travels directly. A small r means few neighbours; a large
r means many.

## 4. Step 2: the neighbours' average direction

Angles cannot safely be averaged arithmetically. Suppose two birds have headings 1° and 359°. Their arithmetic
mean is

$$\frac{1^\circ + 359^\circ}{2} = 180^\circ,$$

which is completely wrong: both birds are pointing almost due east.

Instead, we represent a heading α by the unit vector, or complex number,

$$e^{i\alpha} = \cos\alpha + i\sin\alpha.$$

For particle i we add the vectors of its neighbours,

$$\sum_{j\in\mathcal N_i} e^{i\alpha_j(t)},$$

and take the direction of the sum,

$$\arg \sum_{j\in\mathcal N_i} e^{i\alpha_j(t)}.$$

That is the **circular mean** direction. For 1° and 359° it gives 0°, as it should.

## 5. Step 3: add angular noise

The next heading is

$$\alpha_i(t+1) = \arg \sum_{|\mathbf x_j - \mathbf x_i|\le r} e^{i\alpha_j(t)} + \xi_i,
\qquad \xi_i \sim U[-\eta/2,\ \eta/2].$$

The random angular perturbation ξᵢ is uniformly distributed, with width η, and drawn independently for every
particle and every step.

If η is small, particles closely follow their neighbours. If η is large, their headings fluctuate much more
strongly. Schematically,

$$\text{alignment} \leftrightarrow \text{noise}.$$

Alignment creates order; noise destroys it.

## 6. Step 4: move the particle

After choosing its new heading $\alpha_i(t+1)$, the particle moves exactly a distance v₀ along it:

$$\mathbf x_i(t+1) = \mathbf x_i(t) + v_0\begin{pmatrix}\cos\alpha_i(t+1)\\ \sin\alpha_i(t+1)\end{pmatrix},$$

with periodic boundary conditions. Thus

$$|\mathbf x_i(t+1) - \mathbf x_i(t)| = v_0$$

for one microscopic simulation step.

## 7. Periodic boundary conditions

The box has no physical walls. If a coordinate passes L, it re-enters through the opposite side. For example,
in one dimension with L = 10, x = 9.99 followed by a displacement of +0.03 gives x′ = 10.02, which modulo 10 is
x′ = 0.02. Hence the notation

$$\mathbf x_i(t+1) = \mathbf x_i(t) + \cdots \pmod L.$$

This avoids wall effects. Physically, the simulation domain behaves like the surface of a torus.

## 8. The order parameter

The standard measure of how aligned the flock is, is the polarization or order parameter

$$\varphi = \Big|\frac1N\sum_{i=1}^N e^{i\alpha_i}\Big|, \qquad 0 \le \varphi \le 1.$$

Why between 0 and 1? Every term $e^{i\alpha_i}$ has magnitude 1.

If all headings are identical, $\alpha_1 = \alpha_2 = \dots = \alpha_N = \alpha$, then

$$\frac1N\sum_i e^{i\alpha_i} = e^{i\alpha}, \qquad\text{so}\qquad \varphi = 1:$$

perfect alignment. If the directions are spread approximately uniformly around the circle, the vectors
cancel, $\sum_i e^{i\alpha_i} \approx 0$, and φ ≈ 0. Thus

$$\varphi \approx 1 \Rightarrow \text{strong alignment}, \qquad \varphi \approx 0 \Rightarrow \text{disorder}.$$

In the data we cannot use the headings. The polarization we *measure* is the same formula applied to the
directions of the observed displacements (sections 11 and 15).

<details><summary>The model, compactly</summary>

$$\alpha_i(t+1) = \arg \sum_{|x_j - x_i| \le r} e^{i\alpha_j(t)} + \xi_i, \qquad \xi_i \sim U[-\eta/2,\ \eta/2]$$

$$x_i(t+1) = x_i(t) + v_0\,(\cos\alpha_i(t+1),\ \sin\alpha_i(t+1)) \pmod L$$

The update order is: headings first, from the neighbours at time t, then positions with the new headings.
See [notebook 00](notebooks/00_simulator.ipynb).

</details>

# Part II — What we observe, and what we compute from it

## 9. The key observational problem

The simulator knows every $\mathbf x_i(t)$ and $\alpha_i(t)$. Our hypothetical camera does not. It records
positions only, **once every ten microscopic steps**, so the inference sees

$$\mathbf x_i(0),\ \mathbf x_i(10),\ \mathbf x_i(20),\ \dots$$

rather than every intermediate position. And it never sees $\alpha_i$.

This changes the meaning of several measurements. Over one microscopic step a particle travels a distance v₀.
But over ten steps its net displacement is generally **not** 10 v₀. Why? Because the trajectory bends.

![Six particles over 40 steps: the true path, the recorded positions and the true headings](figures/observation.png)

*What the simulation does and what we keep. Grey: the path of six neighbouring particles, step by step. Black:
the positions we record, one every 10 steps; the arrows join consecutive records, and they are all the data
contain. Red: the true heading αᵢ at each record, which the data do not contain.*

## 10. Net displacement and apparent speed

Suppose a particle moves for ten microscopic steps. Its actual path length is exactly 10 v₀. But the camera
sees only

$$\Delta\mathbf x_i = \mathbf x_i(t+10) - \mathbf x_i(t).$$

Therefore the observed or **apparent speed** is

$$v_{\rm app} = \frac{|\Delta\mathbf x_i|}{10}.$$

By the triangle inequality, with $\Delta\mathbf x_i^{(k)}$ the k-th microscopic step,

$$|\Delta\mathbf x_i| \le \sum_{k=1}^{10}\big|\Delta\mathbf x_i^{(k)}\big|.$$

Every microscopic displacement has magnitude v₀, so $|\Delta\mathbf x_i| \le 10\,v_0$, and therefore

$$v_{\rm app} \le v_0.$$

Equality holds only if all ten microscopic displacements point in exactly the same direction.

**Three examples.** Take v₀ = 1 and a camera that records every two steps.

- *No turning.* Two steps east, (1, 0) and (1, 0). The net displacement is (2, 0), of length 2, so
  $v_{\rm app} = 2/2 = 1 = v_0$. For perfectly straight motion the apparent speed equals the true one.
- *A right-angle turn.* One step east, (1, 0), then one north, (0, 1). The path length is still 2, but the net
  displacement is (1, 1), of length $\sqrt{1^2+1^2} = \sqrt2$, so
  $v_{\rm app} = \sqrt2/2 = 1/\sqrt2 \approx 0.707$, while the true speed is 1. The discrepancy comes purely
  from the turn.
- *A reversal.* One step east, (1, 0), then one west, (−1, 0). The path length is 2, the net displacement
  (0, 0), and $v_{\rm app} = 0$, with the microscopic speed still 1.

So the apparent speed depends not only on the microscopic speed but also on how persistent the trajectory is.
In our flock, at a frame every 10 steps, the apparent speed is 0.96 v₀ at η = 1 and 0.82 v₀ at η = 2
([notebook 01](notebooks/01_observables.ipynb)).

This becomes crucial later, because the noise influences the apparent speed even though v₀ itself is fixed:

$$\text{more angular wandering} \Rightarrow \text{more curved path} \Rightarrow \text{shorter net displacement}
\Rightarrow \text{smaller apparent speed}.$$

Consequently an increase in v₀ can partly compensate an increase in η. That is the origin of one important
parameter ridge (sections 37 and 67), and the two-step examples above contain its whole mechanism.

## 11. Turning angles from three frames

Although the heading αᵢ is hidden, the direction of an observed displacement can be computed:

$$\mathbf u_i(t) = \frac{\Delta\mathbf x_i(t)}{|\Delta\mathbf x_i(t)|}.$$

This is a unit vector along the measured displacement. From three consecutive observations we obtain two
displacement vectors, $\mathbf u_i(t)$ and $\mathbf u_i(t+1)$. Their dot product is

$$\mathbf u_i(t)\cdot\mathbf u_i(t+1) = \cos\Delta\beta_i,$$

where $\Delta\beta_i$ is the angle between the two observed motions. So a natural turning statistic is
$\cos\Delta\beta_i$:

- values near 1 mean $\Delta\beta_i \approx 0$, nearly straight motion;
- values near 0 mean large turns;
- values near −1 mean an almost complete reversal.

We will find that this apparently inexpensive statistic is extremely valuable for identifying the noise.

## 12. Why hidden headings are not fatal

One might first think: since the rule operates on the heading αᵢ, and headings are unobserved, recovering the
rules must be impossible.

But positions contain indirect information about headings. Over a short interval, $\Delta\mathbf x/\Delta t$
tells us about the direction of motion. Across successive intervals, $\Delta\mathbf x(t)$ and
$\Delta\mathbf x(t+\Delta t)$ tell us about turning. The displacement directions of neighbouring particles tell
us about local alignment.

Thus hidden microscopic variables leave statistically recoverable signatures in the observable trajectories.
This is a very general idea in physics:

> hidden states can be inferred through their effects on observables.

## 13. Why use summary statistics?

A complete trajectory contains an enormous number of coordinates. Rather than compare every coordinate
directly, we compress each run into a vector of statistics,

$$\mathbf s = (s_1, s_2, \dots, s_d).$$

These are called **summary statistics**, or simply summaries. Each is an observable computed frame by frame and
averaged over the run. For the main fit d = 7, and the seven belong to three physical groups.

## 14. Group I: individual motion

**Apparent speed.** It measures how far a particle moves between recorded frames (section 10). It is strongly
informative about v₀ (and, through turning, about η).

**Turning angle.** The mean of $\cos\Delta\beta_i$ (section 11). It measures the persistence, or bending, of
individual trajectories, and it is particularly informative about η.

## 15. Group II: collective alignment

We use:

- the **polarization** φ of the observed displacements, $\varphi = \big|\frac1N\sum_i \mathbf u_i\big|$;
- the **velocity correlation** for pairs of particles closer than 1;
- the **velocity correlation** for pairs at distance 1 to 2.

The correlation between the directions of two particles i and j is $\mathbf u_i\cdot\mathbf u_j$. If they
move in the same direction, $\mathbf u_i\cdot\mathbf u_j \approx 1$; if they are unrelated, the average is
about 0; if they move in opposite directions, $\mathbf u_i\cdot\mathbf u_j \approx -1$. The summary averages
this over all pairs whose distance $d_{ij}$ falls in a band [a, b):

$$C_{[a,b)} = \frac{\sum_{i\ne j} \mathbf u_i\cdot\mathbf u_j\ \mathbb 1[a \le d_{ij} < b]}
{\sum_{i\ne j} \mathbb 1[a \le d_{ij} < b]}.$$

By separating pairs according to their distance we probe the spatial scale of alignment. That makes these
measurements potentially informative about r.

## 16. Group III: flock geometry

The two geometric summaries are:

- the **number of clusters**;
- the **nearest-neighbour distance**.

For the cluster count, two particles are linked when their distance is below a threshold of 1. A cluster is a
connected component of this neighbour graph: a group of particles joined by chains of links.

Note an important subtlety:

$$\text{measurement threshold } 1 \ \ne\ \text{model radius } r.$$

The threshold is chosen by us, in the coordinates of the data. It happens that the simulated truth has r = 1,
but the statistic does not rely on knowing this beforehand. A statistic must not use what we are trying to
infer. The same holds for the distance bands of the correlations.

We also compute three more summaries, used only for comparison: the fluctuations of φ (called χ), the velocity
correlation at distances 2 to 4, and the size of the largest cluster.

## 17. Why averaging over time does not eliminate the noise

One might think that a 5000-step run makes every time average extremely precise. That would be true if every
frame were statistically independent. But successive states of a dynamical system are correlated.

Consider a geometric observable $G_t$. If $G_t$ is large, $G_{t+1}$ will probably be large too, because the
configuration of particles cannot rearrange instantaneously. Thus

$$\mathrm{Cov}(G_t, G_{t+k}) \ne 0.$$

A run of M frames does not contain M independent pieces of information. We measure the memory with the
integrated autocorrelation time τ, in frames: a run holds about M/τ independent looks. The speed (τ ≈ 2.5) and
the turning angle (τ ≈ 7) are nearly fresh in every frame. The geometric summaries remember their state for
roughly 26–72 frames, which leaves only about 7–19 effectively independent samples in a 500-frame run.

This explains why slowly varying geometric quantities stay noisy even in apparently long time series.

# Part III — Which observable sees which rule

## 18. A response must be judged against the noise

Suppose changing a parameter θ_q by 10% changes a summary s by Δs. One might conclude: a large Δs means the
statistic is informative. Not necessarily.

Imagine Δs = 0.1.

- Case A: sd(s) = 0.01. The change is enormous compared with the natural fluctuations.
- Case B: sd(s) = 2. The same shift is effectively invisible.

So the relevant quantity is

$$\frac{\text{signal}}{\text{noise}}.$$

## 19. The sensitivity z

We define

$$z = \frac{0.1\ \partial s/\partial\log\theta_q}{\mathrm{sd}(s)}.$$

Let us unpack it. Because

$$d\log\theta = \frac{d\theta}{\theta},$$

a change $\Delta\log\theta \approx 0.1$ corresponds to about a 10% fractional change of θ. So

$$0.1\,\frac{\partial s}{\partial\log\theta}$$

approximates the change of s produced by a 10% perturbation of the parameter. Dividing by sd(s) expresses that
change in units of the ordinary run-to-run fluctuations. Hence

- z = 5 means a 10% parameter change moves the expected statistic by five standard deviations;
- z = 0.2 means the shift is tiny compared with ordinary randomness.

In practice the derivative comes from ±10% differences computed with the *same random numbers*, so that the
difference reflects the parameter and not the noise; sd(s) is the spread over 40 independent runs.

For example, if a summary has sd(s) = 0.04 and a 10% increase of a parameter changes its mean by −0.12, then
z = −0.12/0.04 = −3: the summary drops by three run-to-run standard deviations.

<details><summary>The sensitivity in the notebooks</summary>

$$z = \frac{0.1\ \partial s/\partial \log\theta_q}{\mathrm{sd}(s)}$$

where s is a run's time-averaged observable, θ_q one component of θ, the derivative comes from ±10%
differences with the same random numbers, and sd(s) is its spread over 40 independent runs. See
[notebook 01](notebooks/01_observables.ipynb) §4.

</details>

## 20. Why the sign matters

Suppose z = −5. Its magnitude says the sensitivity is strong. Its sign says that increasing the parameter
*decreases* the statistic.

For example, increasing the noise decreases the mean cosine of the turning angle. That makes physical sense:

$$\eta\uparrow \Rightarrow \text{stronger random angular changes} \Rightarrow \text{less persistent trajectories}
\Rightarrow \cos(\text{turning angle})\downarrow.$$

## 21. Interpreting z = 2

A separation of |z| = 2 does not mean that the parameter is determined. Imagine the value of the summary in one
run as one of two approximately Gaussian distributions,

$$S \sim \mathcal N(\mu_1, \sigma^2) \quad\text{or}\quad S \sim \mathcal N(\mu_2, \sigma^2), \qquad
|\mu_2 - \mu_1| = 2\sigma,$$

one for the original rules and one for the rules changed by 10%. z is the distance between the two centres,
measured in widths of the bell. At |z| = 2 the bells still overlap substantially: deciding from one run which
of the two produced it is wrong about one time in six, and a standard test at the 5% level catches the change
only about half of the time. So one run can easily be ambiguous.

That is why we read |z| ≥ 2 as *one run is at the edge of noticing a 10% change*, not as *one run settles it*.
A clear verdict takes |z| of about 3 or more, or several runs. And z judges one observable on its own. Several
observables used together can do much better, or no better if they repeat the same information, which is the
subject of Parts V and VI.

## 22. What the sensitivity map shows

![Sensitivity z of each summary to each rule](figures/sensitivity.png)

*The sensitivity z of each of the seven summaries (rows) to a 10% change of each rule (columns): the shift of
the summary's mean in units of its run-to-run standard deviation for one 5000-step run. Bold marks |z| ≥ 2;
the colour scale is compressed so that the small values stay visible next to the speed row.*

Three observations stand out.

**The speed v₀.** The apparent speed is overwhelmingly sensitive to v₀, with z ≈ 42. The other statistics
barely respond to it at these parameter values, because the particles move little compared with r. There is,
in effect, a specialised speed detector.

**The noise η and the radius r.** Several summaries respond appreciably to both. The turning angle and the
short-range velocity correlation respond to a 10% change of either by more than two standard deviations. This
immediately hints at a problem: η and r might partly imitate one another.

**The geometry is noisy.** Cluster counts and neighbour distances evolve slowly (section 17), and so are less
precise in a single run. This is a general lesson:

$$\text{physically meaningful} \ne \text{statistically efficient}.$$

The map is a good guide, and it is limited by construction: it looks at one summary and one parameter at a
time. It cannot say whether two summaries carry the same information, or whether two parameters can be told
apart. That needs all the summaries together.

# Part IV — A rate from the transient

## 23. A characteristic time of ordering

A common idea in dynamical systems is to extract a characteristic rate. Start the flock disordered: it
eventually becomes ordered, and we can measure *how fast order emerges*. We measure it on the alignment (φ)
and on the geometry (the nearest-neighbour distance).

An exponential law is the obvious candidate,

$$A(t) = A_\infty + (A_0 - A_\infty)\,e^{-t/\tau},$$

with τ a characteristic relaxation time. A fitted τ, however, requires the observed relaxation to stand clearly
above the stochastic fluctuations over the whole fitting window. We wrote the fitting protocol down before
looking at any curve, and that protocol is what showed that, for the alignment, the noise of an average over
five runs sits at the level where the approach is judged to be over: the fitted rate follows the noise. The
quantity that stands out robustly is the half-time.

## 24. The half-time

Suppose an observable starts at $A_0$ and approaches $A_\infty$. The initial gap is $A_\infty - A_0$. Half the
gap is closed at the time $t_{1/2}$ where

$$A(t_{1/2}) = A_0 + \tfrac12(A_\infty - A_0) = \frac{A_0 + A_\infty}{2}.$$

We find roughly $t_{1/2} \approx 90$ steps for the alignment and $t_{1/2} \approx 260$ steps for the geometry.
The headings align quickly; the arrangement of the flock in space reorganises much more slowly.

![Half-time of the approach to order against the speed](figures/half_time.png)

*Half-time of the approach to the steady state, for an alignment observable (φ) and a geometric one (the
nearest-neighbour distance), at two noise levels. Error bars: standard error over the group of runs. Dotted:
a 1/v₀ law, for reference.*

## 25. Why the speed influences the geometric relaxation

To rearrange, particles must physically move to meet new neighbours. A rough encounter time is

$$t_{\rm encounter} \sim \frac{\text{distance scale}}{\text{speed}},$$

so one expects approximately

$$t_{\rm encounter} \propto \frac{1}{v_0}.$$

This is why decreasing v₀ substantially increases the geometric half-time. Halving the speed multiplies the
geometric half-time by a factor 2.4–3, rather than exactly 2: the system is complicated enough that a pure
inverse-speed law is only a reference scaling.

## 26. What the half-time gives, and what the next step adds

The half-time contains genuine physical information, and it depends strongly on v₀. Its reach is limited in
two ways.

- **Precision.** It varies by 30–60% from one group of runs to another, so seeing a 10% change of v₀ in it would
  take about a hundred groups.
- **Data used.** A run contains 5000 steps, and the transient lasts a few hundred; most of the observations are
  not used. And one scalar cannot disentangle three parameters.

A rate answers its own question, *how fast does the flock order*. The next step is more powerful because it
answers ours, *which rules produced these tracks*, with all the data:

> Can all seven summaries, over the whole steady state, be combined into one statistically principled
> objective?

The answer is the synthetic likelihood.

# Part V — The synthetic likelihood

## 27. Ordinary likelihood

Suppose we have observed data D and a model with parameter θ. The **likelihood** is

$$L(\theta) = p(D\mid\theta).$$

The maximum-likelihood estimate is

$$\hat\theta = \arg\max_\theta\ p(D\mid\theta),$$

or equivalently the maximiser of

$$\ell(\theta) = \log p(D\mid\theta).$$

It answers the question:

> Under which parameter value would the observed data be most probable?

For a deterministic model with Gaussian measurement noise this is familiar: least squares is a special case.

## 28. Why the exact likelihood is unavailable

For the Vicsek system, the observations result from

- hidden headings,
- many stochastic updates,
- nonlinear neighbour interactions,
- partial observation,
- temporal subsampling.

So although it is straightforward to write a simulator,

$$\theta \longrightarrow \text{simulated trajectory},$$

there is no convenient formula for

$$p(\text{observed trajectory}\mid\theta).$$

This kind of problem is called **likelihood-free**, or **simulation-based**, inference. The way forward is to
build a likelihood for the summary statistics rather than for the full trajectory.

## 29. Synthetic likelihood

The synthetic likelihood ([Wood, 2010](README.md#references)) builds it by simulation. Let

$$\mathbf s_{\rm obs}$$

be the vector of seven summaries computed from the observed trajectory. For a candidate parameter vector
θ = (η, r, v₀), simulate B independent runs and compute their summaries,

$$\mathbf s_1, \dots, \mathbf s_B.$$

We use B = 40 for the point fit. Estimate their mean,

$$\hat{\boldsymbol\mu}(\theta) = \frac1B\sum_{b=1}^B \mathbf s_b,$$

and their covariance,

$$\hat\Sigma(\theta) = \frac{1}{B-1}\sum_{b=1}^B (\mathbf s_b - \hat{\boldsymbol\mu})(\mathbf s_b - \hat{\boldsymbol\mu})^\top.$$

Now approximate

$$\mathbf s\mid\theta \sim \mathcal N\big(\boldsymbol\mu(\theta),\ \Sigma(\theta)\big).$$

This is the fundamental assumption of the synthetic likelihood: the summaries are Gaussian. Averages over many
frames tend to be close to Gaussian, and the assumption is among the stated limits of the study.

**Same random numbers at every θ.** We reuse the same random numbers for the B runs at every candidate θ
(common random numbers). Then ℓ is a *deterministic* function of θ, which an optimiser can climb; with fresh
random numbers at each evaluation it would be a noisy one.

## 30. The multivariate Gaussian likelihood

Dropping constants that do not depend on θ, the Gaussian log-likelihood is

$$\ell(\theta) = -\tfrac12\,(\mathbf s_{\rm obs}-\hat{\boldsymbol\mu})^\top\hat\Sigma^{-1}(\mathbf s_{\rm obs}-\hat{\boldsymbol\mu})
- \tfrac12\log\det\hat\Sigma.$$

It has two terms, and each has a job.

## 31. First term: a distance that knows the noise

Define

$$\mathbf d = \mathbf s_{\rm obs} - \hat{\boldsymbol\mu}.$$

Then

$$D_M^2 = \mathbf d^\top\hat\Sigma^{-1}\mathbf d$$

is a squared **Mahalanobis distance**. Why not simply use $|\mathbf d|^2$? Because the observables have
different units and different uncertainties. A discrepancy of 0.02 could be enormous for a precise statistic,
and a discrepancy of 0.2 irrelevant for a noisy one. Multiplying by $\Sigma^{-1}$ weights each discrepancy by
its uncertainty automatically.

## 32. Correlations matter too

Suppose two summaries are almost the same measurement. Treating them as independent would count the same
information twice. The covariance matrix prevents this. Consider

$$\Sigma = \begin{pmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2\\ \rho\sigma_1\sigma_2 & \sigma_2^2\end{pmatrix}.$$

If ρ ≈ 1, the two statistics carry strongly overlapping information, and the inverse covariance discounts the
redundant direction: two misses in the same direction count as one piece of evidence.

This is what the sensitivity map could not do, and it is why the synthetic likelihood is more powerful than
seven separate z values.

## 33. Second term: the volume of the cloud

The term

$$-\tfrac12\log\det\Sigma$$

matters too. For a d-dimensional Gaussian, $\sqrt{\det\Sigma}$ is proportional to the volume occupied by the
probability cloud. A model that spreads its probability over an enormous region should not receive the same
density at its centre as a sharply concentrated one. The determinant term penalises excessive dispersion: a θ
whose runs scatter widely cannot win just because its wide cloud reaches the observation.

## 34. A geometric interpretation

For every candidate θ, the simulated summary vectors form a cloud in seven-dimensional summary space. The
observed data give one point, $\mathbf s_{\rm obs}$. A good parameter value has a cloud whose centre lies near
the observed point, relative to the cloud's own shape. If θ is right, the observed point sits near the centre
of its cloud; if θ is wrong, far out in its tail. So inference becomes:

> find the parameter whose simulated cloud best contains the observation.

This is the conceptual heart of the synthetic likelihood.

<details><summary>The synthetic likelihood in the notebooks</summary>

The optimiser is Nelder–Mead in whitened log-parameters, with several starts. The second protocol adds a
shrinkage estimate of the covariance (Schäfer–Strimmer) and a cheap regression pilot (Fearnhead–Prangle) to
choose the starting region and scale the steps. See [notebook 03](notebooks/03_point_fit.ipynb).

</details>

## 35. The fit

The known truth is

$$\theta^* = (2,\ 1,\ 0.03).$$

The synthetic-likelihood fit is

$$\hat\theta = (2.004,\ 1.014,\ 0.0300),$$

within about 1.5% of the truth. Refitting with four other sets of random numbers moves it by less than 1%
(standard deviations 0.3%, 0.8% and 0.1% for η, r and v₀): the simulation noise does not limit the answer.

That is an encouraging validation. But it does not yet answer the deeper question:

> Are these parameters individually determined by the data?

A numerical optimiser always returns numbers. Identifiability requires more (Part VI).

## 36. Parameter-space geometry: valleys and ridges

Imagine a likelihood of two parameters, $\ell(\theta_1, \theta_2)$. A well-identified model looks
approximately like a compact mountain: one narrow peak.

Now suppose an increase of θ₁ can be compensated by an increase of θ₂. Then many combinations give almost the
same likelihood, and the peak stretches into a valley or ridge. Locally we may have something like

$$a\theta_1 - b\theta_2 \approx c.$$

The data then determine the combination $a\theta_1 - b\theta_2$ well, and the individual components poorly.

## 37. The valley between noise and speed

The likelihood of our fit has such a valley, with a physical explanation we already know from section 10.

Increase the noise, η↑. Particles turn more, so over ten microscopic steps their trajectories bend more and
the net displacement shrinks: $v_{\rm app}\downarrow$.

Increase the speed, v₀↑. The microscopic steps get longer: $v_{\rm app}\uparrow$.

The two effects compensate:

$$\eta\uparrow \text{ together with } v_0\uparrow$$

can leave the apparent speed almost unchanged. This produces an elongated, curved valley in the likelihood.

## 38. Why a narrow curved valley makes the search delicate

Picture the likelihood as a long curved ravine. Moving along the ravine changes the likelihood little; moving
sideways changes it dramatically.

A numerical optimiser proposes finite steps in parameter space. If a step is badly oriented, even a step toward
the solution can leave the narrow valley and land in a region of low likelihood. So the search is delicate even
though the signal is strong.

The randomness of the objective, which one might expect to be the obstacle, is small: about one unit of
log-likelihood, while a 10% change of a parameter costs between 40 and 7400 units. What shapes the search is
the geometry. Of 8 starts across the plausible range, 1 reached the top and one more came close. A second
protocol, written down before running it, used what the first one showed: a better covariance estimate, a
cheap regression to choose the starting region, and steps scaled to that regression. It brought 4 of 8 starts
close to the top. Across 32 fits no competing optimum appeared, so the best of several starts is a reliable
estimate. (Proving that the optimum is unique, the criterion we had set, is beyond what several starts can
give.)

![The synthetic likelihood on the (η, r) plane and where the eight fits stopped](figures/likelihood_map.png)

*The synthetic log-likelihood on the (η, r) plane, with v₀ held at its true value 0.03, and the points where
the eight optimisations of the first protocol stopped (projected on this plane: each also has its own v₀, so a
point can lie off the bright region and still be on the valley in three dimensions).*

The valley is itself a result: it returns in Part VI and, whole, in the posterior of Part VII.

# Part VI — Identifiability and the Fisher information

## 39. Identifiability

We now reach perhaps the deepest conceptual issue of the study.

A parameter is **identifiable** from the observations if meaningfully different values produce detectably
different distributions of observations. If

$$p(D\mid\theta_1) \approx p(D\mid\theta_2)$$

for substantially different $\theta_1 \ne \theta_2$, the data cannot reliably distinguish them.

This is not an optimiser problem. It is an **information** problem. No better algorithm can extract
information that the observations do not contain.

## 40. A simple identifiability example

Suppose a hypothetical observable obeys

$$s = \frac{r}{\eta},$$

and we measure s = 2. Possible parameters include

$$(r, \eta) = (2, 1),\ (4, 2),\ (6, 3),$$

and infinitely many others satisfying r = 2η. Neither r nor η is identifiable separately; only the ratio r/η
is. This is an **exact degeneracy**.

In realistic stochastic models the degeneracy is usually approximate rather than exact, producing a narrow
ridge rather than a perfectly flat line.

## 41. The degeneracy between noise and radius

Take the classic **alignment observables**, the ones used to measure how aligned a flock is: the polarization
φ, its fluctuations χ, and the velocity correlations at distances 1–2 and 2–4. (In
[notebook 04](notebooks/04_identifiability.ipynb) this set is called the *φ-type* set; the seven summaries of
the fit are called the *fit set*.) With these alone, there is a strong trade-off between noise and interaction
radius. Physically:

- η↑ destroys alignment;
- r↑ lets each particle align with more neighbours, and strengthens collective alignment.

Thus

$$\text{more noise} + \text{longer interaction range}$$

can produce similar alignment measurements. The resulting correlation between the errors in η and r is

$$\mathrm{corr}(\eta, r) \approx +0.97,$$

extremely close to +1, and the best fit sits *off* the truth, displaced along the ridge. If one observes
alignment alone, the two parameters are strongly entangled.

![The likelihood on (η, r) with the alignment observables only, and with all seven summaries](figures/ridge_two_panels.png)

*The same plane (v₀ at its true value), with the likelihood built from the alignment observables only (left)
and from the seven summaries of the fit (right). The white line is 3 log-units below the maximum. Left, a long
ridge along which more noise is traded for a longer range; right, a narrow peak around the truth.*

The two flocks below lie on that ridge. Their noise differs by a factor 1.6 and their radius by 30%, and to the
eye they look different. Yet in the alignment observables they are only about 3 run-to-run standard
deviations apart. In the turning angle alone they are 16 apart.

![Two flocks along the ridge](figures/phi_ridge_pair.gif)

## 42. Why the turning angle resolves the degeneracy

Alignment is a collective measurement, and both η and r act on collective order. The turning angle probes
something much closer to the individual angular perturbation: noise directly causes local changes in the
direction of a trajectory. Hence the turning statistic responds strongly to η even when r varies.

Geometrically, imagine the alignment data constrain a long ridge,

$$r \approx f(\eta).$$

The turning statistic supplies a second, nearly independent constraint, approximately

$$\eta \approx \eta_0.$$

Their intersection determines both parameters. With all seven summaries the η–r correlation drops to +0.76,
the peak sits on the truth, and r is determined to about ±5% (95%) from a single run of 40 particles.

This is one of the strongest general lessons of the study:

> A useful observable is one that breaks a parameter degeneracy.

It need not be the most visually dramatic quantity.

## 43. Fisher information

The local geometry of the likelihood is described by its curvature. Near a maximum $\hat\theta$, Taylor
expand:

$$\ell(\theta) \approx \ell(\hat\theta) - \tfrac12(\theta - \hat\theta)^\top I\,(\theta - \hat\theta).$$

Here I is the **Fisher information** matrix (in our case, its version for the synthetic likelihood).

- Large curvature: ℓ falls quickly when a parameter changes, so the parameter is tightly constrained.
- Small curvature: ℓ changes little, so the uncertainty is large.

## 44. The Fisher information we use

We work with logarithmic parameters,

$$\boldsymbol\beta = (\log\eta,\ \log r,\ \log v_0).$$

Let $\boldsymbol\mu(\boldsymbol\beta)$ be the expected summary vector, and define the **Jacobian**

$$J = \frac{\partial\boldsymbol\mu}{\partial\log\theta},$$

written out:

$$J = \begin{pmatrix}
\dfrac{\partial\mu_1}{\partial\log\eta} & \dfrac{\partial\mu_1}{\partial\log r} & \dfrac{\partial\mu_1}{\partial\log v_0}\\[2mm]
\vdots & \vdots & \vdots\\[1mm]
\dfrac{\partial\mu_7}{\partial\log\eta} & \dfrac{\partial\mu_7}{\partial\log r} & \dfrac{\partial\mu_7}{\partial\log v_0}
\end{pmatrix}.$$

Column q says how every summary responds to a change of $\log\theta_q$. We then use

$$I = J^\top\Sigma^{-1}J,$$

with J and Σ estimated from 200 runs at the truth (J from ±10% differences with common random numbers). This
equation deserves careful interpretation.

## 45. Deriving the Fisher information, and what it leaves out

Suppose a small parameter displacement $\delta\boldsymbol\beta$. Linearising the expected summaries,

$$\delta\boldsymbol\mu \approx J\,\delta\boldsymbol\beta.$$

The statistical size of this change of the summaries, measured against their noise, is its Mahalanobis
distance:

$$D^2 = \delta\boldsymbol\mu^\top\Sigma^{-1}\delta\boldsymbol\mu.$$

Substitute $\delta\boldsymbol\mu = J\delta\boldsymbol\beta$:

$$D^2 = (J\delta\boldsymbol\beta)^\top\Sigma^{-1}(J\delta\boldsymbol\beta).$$

Using $(J\delta\boldsymbol\beta)^\top = \delta\boldsymbol\beta^\top J^\top$,

$$D^2 = \delta\boldsymbol\beta^\top\,J^\top\Sigma^{-1}J\,\delta\boldsymbol\beta.$$

Therefore

$$I = J^\top\Sigma^{-1}J.$$

So the Fisher information measures how strongly a change in parameter space moves the expected observables,
relative to their noise. It is the multivariate, correlation-aware version of the sensitivity z: the whole
sensitivity map, with its noise and its correlations, in one matrix.

**What this leaves out.** The argument above uses only how the *means* of the summaries move. For a Gaussian
whose covariance also depends on θ, the full Fisher information has a second term, carried by how the *spread*
of the summaries changes:

$$I_{ab} = \partial_a\boldsymbol\mu^\top\,\Sigma^{-1}\,\partial_b\boldsymbol\mu
+ \tfrac12\,\mathrm{tr}\!\left(\Sigma^{-1}\,\partial_a\Sigma\,\Sigma^{-1}\,\partial_b\Sigma\right).$$

We use the first term only. The second is never negative (it is a positive semidefinite matrix), so leaving it
out can only make I smaller and the error bars wider: our Fisher error bars err on the cautious side.

## 46. Why the inverse of I gives the uncertainty of the estimate

$I^{-1}$ is the approximate **covariance of the estimate** from one run:

$$\mathrm{Cov}(\log\hat\theta) \approx I^{-1}.$$

This is a statement about repeated experiments: if the experiment were repeated many times, the fitted
log-parameters would scatter around the truth with this covariance. It is the smallest scatter an unbiased
estimator can have (the Cramér–Rao bound), and the scatter maximum likelihood reaches when the information is
large.

The quadratic expansion of section 43 shows the same matrix from the side of the likelihood: near its peak,
$\exp\ell$ has the shape of a Gaussian,

$$\exp\big[-\tfrac12(\boldsymbol\beta - \hat{\boldsymbol\beta})^\top I\,(\boldsymbol\beta - \hat{\boldsymbol\beta})\big],$$

and comparing with a multivariate normal density $\propto \exp[-\tfrac12(\mathbf x-\boldsymbol\mu)^\top C^{-1}(\mathbf x-\boldsymbol\mu)]$
gives $C^{-1} = I$, that is $C = I^{-1}$. With a flat prior, the Bayesian posterior of Part VII has this same
covariance near a sharp peak. The two readings coincide in the numbers but answer different questions: the
scatter of an estimator over repeated data, or the plausibility of θ given the one run we have. We keep them
apart, and come to the second in Part VII.

The diagonal of $I^{-1}$ gives the variances. For example,

$$\mathrm{se}(\log r) = \sqrt{(I^{-1})_{rr}}.$$

## 47. Why log-parameter errors are useful

Suppose se(log r) = 0.023. For small errors,

$$d\log r = \frac{dr}{r},$$

so 0.023 corresponds to about a 2.3% relative uncertainty at one standard deviation. A rough 95% Gaussian
interval is

$$\pm 1.96\times 2.3\% \approx \pm 4.5\%.$$

Hence the statement that r is determined to about ±5% at the 95% level from one simulated run. Working in
logarithms makes every standard error a relative error, comparable across parameters of very different size.

## 48. Error correlations

Given the covariance matrix $C = I^{-1}$, the correlation between the errors of parameters a and b is

$$\rho_{ab} = \frac{C_{ab}}{\sqrt{C_{aa}C_{bb}}}.$$

- ρ ≈ +1: the errors tend to move in the same direction.
- ρ ≈ −1: one increases when the other decreases.
- Either case signals a strong ridge.
- ρ ≈ 0: the uncertainties are much less coupled.

For example, two parameters with variances 0.04 each (standard deviations 0.2) and covariance 0.038 have
ρ = 0.038/(0.2 × 0.2) = 0.95. A scatter plot of their joint uncertainty is a long, narrow, positively sloped
ellipse.

## 49. Leave-one-summary-out analysis

A particularly informative experiment: compute the Fisher information with all seven summaries; then remove one
summary, recompute it, and compare the uncertainties. If removing a statistic makes a parameter's uncertainty
explode, that statistic was carrying crucial information. If essentially nothing changes, the statistic was
redundant.

| summaries | se(log η) | se(log r) | se(log v₀) | corr(η, r) |
|---|---|---|---|---|
| all seven | 0.016 | 0.023 | 0.0055 | +0.76 |
| alignment observables only | 0.178 | 0.128 | 2.0 | **+0.97** |
| without the turning angle | 0.032 | 0.041 | 0.012 | **+0.93** |
| without the short-range correlation | 0.017 | 0.031 | 0.0056 | +0.66 |
| without the apparent speed | 0.027 | 0.023 | 0.41 | +0.47 |
| without the polarization | 0.017 | 0.024 | 0.0056 | +0.76 |

(se is the standard error of the logarithm, so 0.023 means about 2.3%.)

## 50. Removing the apparent speed

Without the apparent speed,

$$\mathrm{se}(\log v_0):\ 0.0055 \longrightarrow 0.41.$$

That is a catastrophic loss of information. It confirms what the sensitivity map already suggested: the
apparent speed is essentially the only carrier of v₀.

## 51. Removing the turning angle

Without the turning angle, the η–r correlation rises from about 0.76 to 0.93, and the errors of η and r nearly
double. So the turning angle does a qualitatively important job: **it breaks the noise–radius degeneracy.**

## 52. Removing the polarization

Removing the polarization, by contrast, barely changes the errors. This does not mean the polarization is
useless. It means that **its information is already carried by the other summaries.**

This distinction is essential. A statistic may be highly sensitive to a parameter by itself and still be
redundant once other, correlated observables are included. The polarization, the observable a physicist
reaches for first, is redundant here.

## 53. Information grows as signal squared over noise squared

In the simplest case, one statistic and one parameter, everything is a scalar. Take a statistic whose mean
responds with $\partial\mu/\partial\log\theta = 4$ and whose standard deviation is σ = 2. Then J = 4,
Σ = σ² = 4, and

$$I = J\,\Sigma^{-1}J = 4\cdot\tfrac14\cdot 4 = 4, \qquad I^{-1} = \tfrac14, \qquad \mathrm{se}(\log\theta) = \tfrac12:$$

a rather weakly constrained parameter. Now keep the response but halve the noise, σ = 1:

$$I = 4\cdot 1\cdot 4 = 16, \qquad I^{-1} = \tfrac1{16}, \qquad \mathrm{se}(\log\theta) = \tfrac14.$$

Halving the noise of the measurement halves the standard error of the parameter. That is what the Fisher
information formalises:

$$\text{parameter information} \sim \frac{\text{signal}^2}{\text{noise}^2}.$$

<details><summary>The Fisher information in the notebooks</summary>

J and Σ are estimated from 200 runs at θ*; summaries that are constant across runs (such as the number of
clusters in a dense box) are dropped, since they carry no information and would make Σ singular. The table of
section 49 removes one summary at a time from J and Σ. See [notebook 04](notebooks/04_identifiability.ipynb).

</details>

# Part VII — From curvature to the full uncertainty

## 54. What the curvature assumes

The Fisher information treats the log-likelihood near its maximum as quadratic,

$$\ell(\theta) \approx \ell(\hat\theta) - \tfrac12(\theta - \hat\theta)^\top I\,(\theta - \hat\theta),$$

which corresponds to a locally Gaussian region of uncertainty. It gives error bars quickly, but the actual
likelihood may be curved, skewed, asymmetric, or slightly rough because it is estimated by simulation.

The next step asks the question more directly:

> What happens if we repeatedly generate new datasets and refit them?

That is the bootstrap.

## 55. Parametric bootstrap

A parametric bootstrap works like this. Start from the fitted model $\hat\theta$.

1. Pretend $\hat\theta$ is the true parameter.
2. Generate a new artificial dataset, $D_1^* \sim p(D\mid\hat\theta)$.
3. Fit that dataset exactly as the original: $\hat\theta_1^* = \mathrm{fit}(D_1^*)$.
4. Repeat, for $D_2^*, D_3^*, \dots, D_B^*$.

This yields $\hat\theta_1^*, \hat\theta_2^*, \dots, \hat\theta_B^*$. The spread of these estimates approximates
the sampling uncertainty of the original estimator.

## 56. What the bootstrap adds to the curvature

The Fisher information asks:

> How curved is the likelihood near this fitted point?

The bootstrap asks:

> If nature repeated the experiment many times under approximately these rules, how much would our inferred
> parameters change?

It therefore includes more of the estimator's actual behaviour, including the optimiser, and it can test the
fitting procedure itself. We ran it with 30 new datasets and a control fixed in advance: five of the datasets
refitted from a second starting point, whose two fits should agree.

The control is what made this step informative. The two fits of a dataset differed by about one standard
error, and the reason turned out to be a property of the problem, explained in the next two sections.

## 57. Two different scales of likelihood variation

On large scales, the likelihood distinguishes parameter values strongly: a 10% change of a parameter costs
tens, hundreds or thousands of log-likelihood units.

Uncertainty intervals, however, are about much smaller changes. At the scale of one standard error, the true
differences of likelihood are only about one unit. And the likelihood estimated by simulation has small-scale
variation of about that same size: the jitter that a chaotic simulator leaves even with fixed random numbers.
So we have

> a likelihood smooth at the scale of the fit, and rough at the scale of the uncertainty.

This is why the exact position of the maximum fluctuates.

## 58. Why maximising inherits the roughness

Write the computed likelihood as

$$\ell_{\rm computed}(\theta) = \ell_{\rm ideal}(\theta) + \epsilon(\theta),$$

where ε is the small, simulation-induced roughness. To compare very different parameter values,

$$|\ell_{\rm ideal}(\theta_1) - \ell_{\rm ideal}(\theta_2)| \gg |\epsilon|,$$

and the roughness hardly matters. But the maximiser is located where $\ell_{\rm computed}$ is microscopically
largest, and a small perturbation can move that location substantially along a flat direction. So any method
that *maximises* the likelihood for each dataset inherits about one standard error of scatter from ε.

A second version of the bootstrap, which smooths the likelihood with a fitted surface before maximising,
flagged a different and fixable problem in its own controls. The two versions agree on the size of the error
bars as orders of magnitude. To go further, a method must not depend on the exact location of a maximum. That
is what the next tool provides.

## 59. Bayesian inference

Bayesian inference starts from Bayes' theorem,

$$p(\theta\mid D) = \frac{p(D\mid\theta)\,p(\theta)}{p(D)}.$$

Ignoring the normalisation constant,

$$p(\theta\mid D) \propto p(D\mid\theta)\,p(\theta),$$

where

- $p(D\mid\theta)$ is the likelihood,
- $p(\theta)$ is the prior,
- $p(\theta\mid D)$ is the posterior.

Instead of returning only $\hat\theta$, the posterior assigns a probability density across the whole plausible
region of parameters. We use a prior uniform in log θ over the box of plausible values in which the fits were
started.

The posterior is more powerful than a point estimate for exactly the reason of section 58: it does not
maximise. It *averages* over the likelihood, so the small-scale roughness matters little.

## 60. Why a posterior is particularly useful for ridges

Suppose the region of high likelihood is an elongated diagonal strip. A point estimate reports one spot
somewhere inside it, and hides the main structure. Samples of the posterior, instead, fill the whole
correlated region, so we can see directly statements such as

$$\theta_1\uparrow \iff \theta_2\uparrow.$$

This is exactly what we will see for η and v₀ (section 67).

## 61. MCMC

The posterior usually cannot be sampled directly. A **Markov chain Monte Carlo** algorithm builds a sequence

$$\theta^{(1)}, \theta^{(2)}, \theta^{(3)}, \dots$$

whose long-run distribution is the posterior. A basic Metropolis–Hastings step is:

1. from the current state θ,
2. propose θ′,
3. compute an acceptance probability,
4. either move to θ′ or stay at θ.

For a symmetric proposal, the acceptance probability is

$$a = \min\left[1,\ \frac{p(D\mid\theta')\,p(\theta')}{p(D\mid\theta)\,p(\theta)}\right].$$

After enough steps, the frequency with which the chain visits a region reflects its posterior probability.

We walk in log θ, and propose θ′ with a Gaussian step shaped by the Fisher covariance of Part VI: the steps are
about one standard error long and correctly correlated, so the chain moves along the ridges instead of
across them.

## 62. A likelihood that is itself simulated

Ordinary MCMC assumes we can evaluate $L(\theta) = p(D\mid\theta)$. Here only a simulation-based estimate
$\hat L(\theta)$ is available. That seems dangerous: if every step sees a slightly different random
likelihood, why should the chain target the right distribution? This is where pseudo-marginal MCMC enters.

## 63. The pseudo-marginal principle

The key result ([Andrieu & Roberts, 2009](README.md#references)) is this. If the likelihood estimator is

- **positive**, and
- **unbiased**,

$$\mathbb E\big[\hat L(\theta)\big] = L(\theta),$$

and the estimate at the current point is kept, not recomputed, until the chain moves, then the
Metropolis–Hastings chain still targets the posterior of L(θ). The noise in the likelihood slows the sampling
(after accepting a point whose likelihood was overestimated by luck, the chain sticks there for a while), but
it does not change the distribution the chain converges to.

## 64. Which posterior is "exact"?

The original trajectories do not have a tractable likelihood. The analysis first makes the approximation

$$\mathbf s\mid\theta \approx \mathcal N\big(\boldsymbol\mu(\theta), \Sigma(\theta)\big).$$

That Gaussian model of the summaries is the synthetic likelihood. Pseudo-marginal MCMC targets the posterior
of *that* construction exactly. It does not remove the initial Gaussian approximation. So there are two
levels:

$$\text{real trajectory process}\ \longrightarrow\ \text{Gaussian synthetic likelihood of the summaries}
\ \longrightarrow\ \text{posterior computed exactly for that approximation}.$$

Keeping the two levels apart is what prevents overclaiming: the chain samples the posterior of the Gaussian
approximation without any further approximation, not the posterior of the true likelihood of the tracks.

## 65. Why the plain synthetic likelihood is biased

The Gaussian likelihood of the summaries depends on μ and Σ. These are estimated from finitely many
simulations, $\hat{\boldsymbol\mu}$ and $\hat\Sigma$. Plugging the estimates into

$$\mathcal N(\mathbf s_{\rm obs};\ \hat{\boldsymbol\mu}, \hat\Sigma)$$

does not, in general, give an unbiased estimate of $\mathcal N(\mathbf s_{\rm obs};\ \boldsymbol\mu, \Sigma)$.
Symbolically,

$$\mathbb E\big[f(\hat{\boldsymbol\mu}, \hat\Sigma)\big] \ne f\big(\mathbb E[\hat{\boldsymbol\mu}],\ \mathbb E[\hat\Sigma]\big).$$

The nonlinearity causes the bias. With 40 simulations per evaluation, the plain synthetic likelihood comes out
up to roughly twice too sharp in our problem: overconfident.

The **unbiased synthetic likelihood** ([Price, Drovandi, Lee & Nott, 2018](README.md#references)) replaces the
plug-in density by an estimator that is exactly unbiased for the Gaussian density when the number of
simulations n exceeds d + 3, d being the number of summaries ([Ghurye & Olkin, 1969](README.md#references); we
use the formula as written in [An, South & Drovandi, 2022](README.md#references)). We use n = 30 runs per
estimate, chosen by a rule fixed in advance, and fresh random numbers at every proposal, which turns the
small-scale roughness into noise the chain averages out.

## 66. Posterior results

Six chains, started apart, agree, and nothing was tuned after looking at them.

| | mean | 95% interval | truth |
|---|---|---|---|
| η | 2.014 | [1.965, 2.061] | 2 |
| r | 1.009 | [0.985, 1.038] | 1 |
| v₀ | 0.0301 | [0.0298, 0.0303] | 0.03 |

All three true values fall inside the corresponding intervals.

![Posterior samples for each pair of parameters](figures/posterior.png)

*Samples of the posterior from the six MCMC chains after burn-in, one pair of parameters per panel (every
fourth sample shown). The thin straight cloud in the middle panel is the η–v₀ ridge.*

The posterior standard deviations are 0.6–0.7 of the Fisher ones of Part VI. Part of the difference comes from
the ±10% differences used for J, which slightly underestimate the slopes (with ±3% the Fisher errors shrink by
8–15%); the covariance term left out of I (section 45) acts in the same direction.

<details><summary>The posterior in the notebooks</summary>

Random-walk Metropolis–Hastings in log θ, proposal covariance from notebook 04's Fisher matrix, prior uniform
in log θ on notebook 03's box, n = 30 simulations per unbiased estimate, six chains. Acceptance is 14–16% in
every chain, as expected with a noisy likelihood. See [notebook 06](notebooks/06_posterior.ipynb).

</details>

## 67. The posterior correlation between noise and speed

The posterior has

$$\mathrm{corr}(\eta, v_0) \approx +0.98,$$

extremely strong. Posterior draws approximately satisfy

$$\eta\uparrow \Longrightarrow v_0\uparrow.$$

The physical mechanism is again that of section 10:

$$\eta\uparrow \Rightarrow \text{more turning} \Rightarrow v_{\rm app}\downarrow, \qquad
v_0\uparrow \Rightarrow v_{\rm app}\uparrow.$$

So the measured trajectories constrain a *combination* of η and v₀ very tightly. They are known much better as
a pair than one by one. This is the same valley that made the search of section 38 delicate, now seen whole.

# Part VIII — What carries over to other problems

Parts I–VII climbed the ladder on one flock. This part steps back: what did the climb teach that applies to any
inverse problem of this kind, and what did it cost? Where an idea was already developed, we only point back to
it.

## 68. Three things that are easy to confuse

Three properties of a parameter are often taken for one another. The flock separates them cleanly.

**Sensitivity:** does changing the parameter move the data? (Part III.) Both η and r move the alignment
observables strongly.

**Precision:** how well is a summary measured? More runs, or a longer run, make it more precise.

**Identifiability:** can the effects of different parameters be told apart? (Section 39.)

The first two do not give the third. η and r are both strongly sensitive, and the alignment summaries can be
measured as precisely as we like; with those summaries alone, the two still trade off along a ridge
(section 41). Infinite precision only makes the ridge thinner, not shorter: in the example of section 40 we
could know r/η to every decimal and still know neither r nor η.

The Fisher information shows where the problem lives. Each parameter is a column of J, its "fingerprint" on the
summaries. If two fingerprints are nearly parallel, $\partial\boldsymbol\mu/\partial\eta \approx
c\,\partial\boldsymbol\mu/\partial r$, then $I = J^\top\Sigma^{-1}J$ has an eigenvalue close to zero. Its
eigenvector is the direction of the ridge, and the uncertainty along it, $1/\lambda$, is huge. That is how we
placed the two flocks of section 41: exactly along that eigenvector.

## 69. Better observables, not more of them

If more data of the same kind cannot break a ridge, what can? An observable whose fingerprint points in a
**new direction**.

The smallest example has two unknowns, a and b. Measuring $s_1 = a + b = 10$ leaves a whole line of answers:
(1, 9), (5, 5), (9, 1)… Measuring $s_1$ ten more times does not help. A second observable that responds
*differently*, $s_2 = a - b = 2$, crosses that line at one point: a = 6, b = 4.

The turning angle is the flock's $s_2$. The alignment summaries all say roughly "η − r", and adding more of
them, or adding the polarization to the others, only repeats it (section 52). The turning angle says roughly
"η alone", and that is what pins both down (section 42). When designing an experiment, the question to ask of
a candidate observable is not "how much does it react?" but "does it react *differently* from what I already
measure?"

## 70. Precise can beat varied: the dense box

One result went against our intuition, and it completes the previous section. We expected a dense box, where
the 40 particles form a single cluster, to be worse at determining r: its geometry is dead, since the number of
clusters is always one. It is better: se(log r) = 0.015, against 0.023 in the sparse box.

The reason is the balance of section 53, information ∼ sensitivity² / variance. In the dense box the flock is
one ordered group, and its runs barely differ from one another, so even small responses of the alignment
summaries to r stand far above the noise. The sparse box offers more *kinds* of information; the dense box
offers more *precise* information. Neither "more variety" nor "more precision" wins in general: what counts is
signal against noise, direction by direction.

## 71. The camera is part of the model

How we observe shapes what can be inferred. Had the camera recorded every step, each displacement would have
length exactly v₀ and point exactly along the heading: v₀ and the noise could be read off almost directly, and
the inverse problem would be nearly trivial. Recording every ten steps blurs both together through the
apparent speed (section 10), and that blur is where the η–v₀ valley and the +0.98 posterior correlation come
from (sections 37 and 67).

We chose ten steps deliberately: short enough that turning stays visible, long enough that each displacement
mixes several hidden headings, as a real camera would. The general point is that the observation model (what
is recorded, how often, with what noise) belongs to the model, and changes which questions the data can
answer.

## 72. A number is not a result

Because the data are synthetic, we could check that the fit landed within 1.5% of the truth. With real data
there is no truth to compare with, and a fit with many decimals, r = 1.013742, may mean r = 1.0 ± 0.4. An
optimiser returns numbers whether or not the data determine them. What makes a result is the estimate together
with its uncertainty and with the identifiability behind it; Parts VI and VII exist to supply those.

## 73. What it costs

Simulation-based inference is nested, and its cost multiplies:

$$\text{an MCMC step} \ \supset\ \text{a likelihood estimate} \ \supset\ \text{30–40 complete simulations}.$$

A fit needs many likelihood evaluations, the bootstrap needs a fit per new dataset, and the posterior needs
thousands of estimates: the six chains took about 12 hours on 6 cores. This is why the repository keeps the
results of the long runs in `results/`, so that every notebook opens and runs in minutes.

## 74. The ladder at a glance

| step | question | tool | what it adds to the step before |
|---|---|---|---|
| 1 | Which summaries react to which rule? | sensitivity z | a map, in units of each summary's own noise |
| 2 | Can one number from the dynamics tell us the rules? | half-time | a well-defined physical time, strongly dependent on v₀ |
| 3 | Can we fit all three rules at once? | synthetic likelihood | the whole steady state and all summaries, weighted by noise and correlations |
| 4 | Which combinations do the data determine? | Fisher information | error bars, correlations, ridges, and what each summary carries |
| 5 | How would the estimate change with new data? | parametric bootstrap | the estimator's real scatter, and the roughness of the likelihood |
| 6 | What is the whole distribution of plausible rules? | unbiased synthetic likelihood + pseudo-marginal MCMC | the full shape of the uncertainty, free of the location of a maximum |

## 75. What the study shows, and what it does not

**It shows**, on synthetic data, that positions alone, one frame every ten steps, determine the three Vicsek
rules to a few percent; that the information is unevenly spread (the apparent speed carries v₀, the turning
angle breaks the η–r ridge, the polarization is redundant); and that the remaining correlations, such as η–v₀,
are part of the answer and have a physical reading.

**It does not show** that the method works on real flocks. The data are simulated, without tracking noise; the
box is periodic, without walls or collisions; there are 40 particles at one design point; the angular noise is
uniform; the summaries are assumed Gaussian.

**And it solves the easier of two inverse problems.** It is *parameter estimation*: given that the rules have
the form of the Vicsek model, what are (η, r, v₀)? The harder problem is *model selection*: is the alignment
metric or topological, does turning have inertia, is there attraction and repulsion? There the form of the rules
is itself unknown. The tools here are the first rung of that harder problem, not a solution to it.

## 76. The whole chain

Running the flock backwards, in one line each:

$$\theta \to \{\alpha_i(t)\} \to \{\mathbf x_i(t)\} \to D_{\rm obs} \to \mathbf s_{\rm obs}$$

is how nature, or the simulator, produces the data: rules, hidden headings, tracks, what the camera keeps, its
summaries. Inference runs the other way:

$$\theta \xrightarrow{\ \text{simulate}\ } \hat{\boldsymbol\mu}(\theta),\ \hat\Sigma(\theta)
\xrightarrow{\ \text{compare with } \mathbf s_{\rm obs}\ } \hat L(\theta)
\longrightarrow \hat\theta,\ \ I,\ \ p(\theta\mid D):$$

a point estimate, the local identifiability, and the full posterior.

## 77. The essential equations on one page

**Vicsek update**

$$\alpha_i(t+1) = \arg\sum_{|\mathbf x_j - \mathbf x_i|\le r} e^{i\alpha_j(t)} + \xi_i, \qquad \xi_i \sim U[-\eta/2,\ \eta/2]$$

$$\mathbf x_i(t+1) = \mathbf x_i(t) + v_0(\cos\alpha_i, \sin\alpha_i) \pmod L$$

**Polarization**

$$\varphi = \Big|\frac1N\sum_i e^{i\alpha_i}\Big|$$

**Apparent speed**

$$v_{\rm app} = \frac{v_0}{10}\Big|\sum_{k=1}^{10} e^{i\alpha_k}\Big| \le v_0$$

**Sensitivity**

$$z = \frac{0.1\ \partial s/\partial\log\theta_q}{\mathrm{sd}(s)}$$

**Synthetic likelihood**

$$\ell(\theta) = -\tfrac12(\mathbf s_{\rm obs}-\hat{\boldsymbol\mu})^\top\hat\Sigma^{-1}(\mathbf s_{\rm obs}-\hat{\boldsymbol\mu}) - \tfrac12\log\det\hat\Sigma$$

**Local response of the summaries**

$$\delta\boldsymbol\mu \approx J\,\delta\log\theta$$

**Fisher information (mean term)**

$$I = J^\top\Sigma^{-1}J$$

**Approximate covariance of the estimate**

$$\mathrm{Cov}(\log\hat\theta) \approx I^{-1}$$

**Bayesian posterior**

$$p(\theta\mid D) \propto L(\theta)\,p(\theta)$$

**Pseudo-marginal requirement**

$$\hat L(\theta) > 0, \qquad \mathbb E\big[\hat L(\theta)\big] = L(\theta)$$

## 78. Six ideas to keep

**1. Forward and inverse are different problems.** Knowing how to go from θ to data does not tell us how to go
from data to θ.

**2. A response matters only relative to the noise.** Always think signal over noise.

**3. Affecting the data is not being identifiable.** Two parameters can leave the same fingerprint.

**4. Better observables beat more observables.** The turning angle matters because it points where the
alignment summaries do not.

**5. A point estimate is not a result.** A fit returns numbers even when the data barely determine them.

**6. Correlation is part of the answer.** η = 2.014 ± … does not describe the inference when η is tied to v₀;
the joint shape says how the mechanisms compensate each other, and that is physics.

That is what the study is about in the end: not merely finding parameters, but determining what the
observations genuinely allow us to know about the rules that generated them.
