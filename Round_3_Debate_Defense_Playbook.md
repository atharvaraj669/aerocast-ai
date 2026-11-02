# AI SANSAD 2026 — ROUND 3: PARLIAMENTARY DEBATE PLAYBOOK
## "Build the Model. Defend the Logic. Shape the Consensus."
### Team: Prompt Diplomats | Team ID: UYU672B3 | Leader: Atharva Sandeep Patil
### Assigned Problem: Hourly Wind Power Generation Forecasting

---

## 1. The Parliamentary Framework & Our Strategic Stance

In Round 3, technical competence is only half the battle; the remaining half is **rhetorical rigor, technical cross-examination, and building policy consensus**.

* **Our Parliamentary Persona**: Pragmatic, Physics-Informed Technocrats. We champion safety, physical reality, computational feasibility, and national grid security over naive "deep learning hype."
* **Core Parliamentary Motto**: *"Algorithms cannot violate aerodynamics. A model that predicts without physics is an existential threat to the grid."*

---

## 2. Anticipated Attack Vectors (Opposing Teams & Jury) & Defense Shields

### Attack 1: *"Why use a hybrid model (LightGBM + Transformer) instead of an end-to-end Pure Deep Learning model like PatchTST, Informer, or LSTM?"*
* **Our Defense Response**:
  > *"Delegate, in academic benchmarks, end-to-end neural networks look glamorous. But in real-world wind farm dispatch, pure deep networks suffer from two fatal vulnerabilities: **poor handling of tabular step-functions** and **high sample inefficiency**. A wind turbine power curve has sharp, non-differentiable phase transitions: at cut-in speed ($3\text{ m/s}$), power immediately jumps; at rated speed ($12\text{ m/s}$), it hits an aerodynamic flat ceiling; and at cut-out ($25\text{ m/s}$), it plunges to zero within seconds.
  > 
  > Tree-based algorithms (LightGBM/CatBoost) segment piecewise linear regimes with surgical precision that gradient-descent neural networks smooth over and miss. By pairing gradient boosting for sharp tabular boundary transitions with a Temporal Fusion Transformer for long-horizon atmospheric inertia, AeroCast-AI captures the best of both worlds with a sub-100 millisecond inference footprint that runs effortlessly on edge hardware."*

---

### Attack 2: *"Why did you spend time on physical feature engineering like the Hellmann wind shear law and air density, rather than letting a deep network learn these representations automatically?"*
* **Our Defense Response**:
  > *"To suggest that a neural network should 'learn' the laws of thermodynamics from a few thousand training rows is like asking a self-driving car to re-discover optics from scratch before reading a stop sign.
  > 
  > The cubic relationship $P \propto v^3$ and the ideal gas law $\rho = \frac{p}{RT}$ have been verified physical facts for centuries. In India's varied topography—from the high ambient temperatures of Rajasthan where air is thin, to coastal Tamil Nadu where air is dense—air density variations cause up to a 15% generation swing at identical wind speeds. Handing the model pre-computed kinetic flux ($\frac{1}{2}\rho v^3$) frees up model capacity to focus on genuine micro-turbulence and wake interactions rather than struggling to rediscover basic physics with limited historical data."*

---

### Attack 3: *"Numerical Weather Predictions (NWP) have massive spatial errors. If your input forecast is wrong, doesn't your model produce garbage?"*
* **Our Defense Response**:
  > *"That is precisely why naive models fail and why AeroCast-AI was engineered with two distinct defensive layers.
  > First, our Temporal Fusion Transformer incorporates **Variable Selection Networks (VSN)** that measure the historical correlation and dynamic reliability of each weather feature, automatically down-weighting noisy forecasts.
  > Second, we reject single-point deterministic forecasting. AeroCast-AI produces **Conformal Probabilistic Prediction Bands ($P_{10}, P_{50}, P_{90}$)**. When NWP forecasts exhibit high variance or rapid barometric pressure swings, the confidence envelope expands dynamically. This warns grid dispatchers of high atmospheric volatility, allowing them to procure adequate spinning reserves rather than being blindsided by a deterministic point forecast error."*

---

### Attack 4: *"What happens during extreme meteorological events—like a cyclone, sudden storm gust, or blade icing?"*
* **Our Defense Response**:
  > *"This is where our **Physics-Constrained Governor** is irreplaceable. An unconstrained machine learning model presented with 28 m/s wind speed will naively extrapolate maximum power generation. In reality, modern turbines feather their blades and engage mechanical brakes at 25 m/s to prevent structural disintegration—generation drops immediately to zero!
  > 
  > AeroCast-AI implements hardcoded aerodynamic cut-out rules and hysteresis logic. When wind velocity breaches the aerodynamic threshold, our model overrides any statistical prediction and enforces zero generation. We refuse to compromise physical safety for statistical elegance."*

---

### Attack 5: *"How do you handle missing sensor telemetry or delayed NWP data in real-time?"*
* **Our Defense Response**:
  > *"AeroCast-AI incorporates a **Graceful Degradation Protocol**:
  > - **Tier 1 (Full Mode)**: SCADA + NWP Multi-Engine Ensemble.
  > - **Tier 2 (NWP Delay Mode)**: If weather forecast feeds lag, the model seamlessly transitions to an autoregressive rolling lag engine trained on local farm SCADA and barometric sensors.
  > - **Tier 3 (Sensor Drop Mode)**: Kalman filter imputation across neighboring turbines in the cluster based on directional spatial adjacency."*

---

## 3. Offensive Cross-Examination Arsenal (Critiquing Competing Teams)

When opposing teams present their solutions during the debate, use these targeted, polite, yet devastating technical cross-examination questions:

### Question to Teams with Pure Deep Learning (LSTM, GRU, Transformers only):
> *"Honorable Delegate, your model relies entirely on recurrent or attention layers without physical constraints. In the event of a sudden 26 m/s storm gust, what stops your neural network from predicting 150% peak power generation, when the physical wind turbine has shut down its brakes for safety? Have you verified whether your model ever outputs negative megawatts?"*

### Question to Teams using Standard K-Fold Cross-Validation:
> *"Distinguished Delegate, could you clarify your cross-validation scheme? If you used standard random K-Fold or train-test splits on continuous time-series data, how did you account for the 48-hour atmospheric autocorrelation leakage between training and testing sets? Did you purge and embargo your splits?"*

### Question to Teams Predicting Only Single-Point Generation:
> *"Respected Delegate, point forecasts have zero utility during grid frequency dips. If your model predicts 45 MW, how does a grid operator know whether the 90% confidence lower bound is 42 MW or 15 MW? How does your architecture quantify tail-risk for national load dispatchers?"*

### Question to Teams Ignoring Wind Direction or Air Density:
> *"Delegate, you passed raw wind direction in degrees [0–360] into your model. How does your model resolve the mathematical discontinuity where 359 degrees and 1 degree are physically adjacent but numerically opposite in distance metrics?"*

---

## 4. The Parliamentary Motion & Consensus Resolution (Draft Bill)

To fulfill the event's challenge to *"Shape the Consensus"*, Team Prompt Diplomats will introduce the following draft resolution during the debate:

### **PARLIAMENTARY RESOLUTION 01/2026**
**Sponsored by:** Team Prompt Diplomats (UYU672B3)  
**Subject:** *A National Standard for Physics-Constrained Probabilistic Forecasting in Renewable Energy Grid Integration*

> **Preamble:**  
> *Recognizing that the Republic of India aims to integrate 500 GW of non-fossil energy by 2030;  
> Noting that renewable intermittency currently imposes over ₹2,000 Crores in grid deviation penalties and forces thermal coal plants to operate sub-optimally as spinning reserves;  
> Acknowledging that black-box artificial intelligence models without physical verification create catastrophic grid vulnerability;*
> 
> **The House Hereby Resolves:**
> 1. **Mandate Physics-Informed Safeguards**: All commercial AI/ML forecasting models deployed for national grid scheduling must incorporate deterministic physical boundary guards (prohibiting negative generation and enforcing turbine cut-out hysteresis).
> 2. **Transition to Calibrated Probabilistic Scheduling**: The Central Electricity Regulatory Commission (CERC) and National Load Despatch Centre (NLDC) shall formally recognize $P_{10}$ to $P_{90}$ quantile envelopes rather than deterministic point schedules for spinning reserve planning.
> 3. **Open Meteorological Standardization**: State Load Despatch Centres (SLDCs) shall mandate standardized, hub-height micro-NWP API feeds to democratize high-accuracy forecasting for small and medium renewable developers.
