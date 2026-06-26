# SunGenie eAnalytiX O&M Agent — Capabilities & Query Guide

This guide highlights the core capabilities of the SunGenie O&M Assistant and provides a catalog of questions/queries it is calibrated to answer.

---

## 🛠️ Core Capabilities

The agent operates as a **telemetry-calibrated solar engineering assistant** directly connected to the Jamnagar Solar Facility's SQLite database containing **204,218 records** across **42 active assets**.

### 1. Robust Asset Resolver
The agent features a regex-based parser that maps loose operator terminology to unique database identifiers (prefixed with `JAMNAGAR_VIRTUAL_GATEWAY_`):
*   **Batteries**: `B1BCT1`, `B1BCT2`, `B1BCT3`, `B2BCT1`, `B2BCT2`, `B2BCT3`, `B3BCT2`
*   **Inverters**: `B1INV1`, `B3INV1`
*   **Meters**: `B1MFM1`, `B1MFM2`, `B2MFM1` through `B2MFM16`, `B3MFM1`, `B3MFM2`, `B3MFM3`
*   **PCS (Power Conversion System)**: `B1PCS1`, `B2PCS1`, `B3PCS1`
*   **PQM (Power Quality Monitor)**: `B1PQM2`, `B2PQM1`, `B3PQM1`, `B3PQM2`
*   **DC Converters**: `B2DCCON1`, `B2DCCON2`, `B2DCCON3`
*   **Weather Stations**: `PPCWMS1`

### 2. Multi-Asset Comparative Analysis
When multiple assets are specified or a type comparison is requested, the agent fetches telemetry for each asset, compares their latest metrics (SOC, active power, voltage, temperature, or efficiency), and automatically plots a comparison chart.

### 3. Integrated ML Diagnostics Pipelines
The assistant is backed by eight analytics pipelines that process telemetry:
*   **Performance Ratio (PR)**: Formatted as a precise decimal (e.g. `0.7788`) against the `0.85` target.
*   **Soiling Rate**: Detects daily dust loss rates using change-point analysis of panel washing cycles.
*   **SCB Outliers**: Identifies loose connections/shading by running statistical Z-score outlier checks.
*   **Thermal hotspot warnings**: Measures NOCT deviations to flag hotspots.
*   **Availability**: Calculates operating hours restricted to daytime daylight windows.

### 4. Primary AI Model & Fallbacks
The intelligence of the O&M agent is orchestrated through a prioritized multi-model calling structure:
*   **Primary Model**: **Google Gemini (Gemma 4 26B)** (specifically using the requested `gemma-4-26b-a4b-it` model) functions as the primary inference engine to provide fast and accurate answers directly from local telemetry.
*   **Fallback Models**: If Google API is unreachable, requests are automatically routed to:
    *   *Nvidia Ising Calibration 35B*: Dedicated specialist for physical calculations, irradiance correlations, and thermal hotspot anomalies.
    *   *Nvidia Gemma 3 12B*: Dedicated engine for operational ticketing classifications and general QA.

---

## ❓ Supported Questions & Queries

Here is a categorized list of sample queries you can type into the assistant:

### 1. General Plant Health & ML Diagnostics
These queries execute background ML pipelines and render interactive charts with info tooltips:

*   *"What is the overall generation performance and PR gap breakdown for the plant?"*
    *(Renders: PR Gap Loss Attribution Bar Chart)*
*   *"Are there any underperforming strings on Inverter B1INV1? If so, generate a warning ticket."*
    *(Renders: SCB String Z-Score Bar Chart)*
*   *"What is the inverter fleet DC-AC conversion efficiency across different load levels?"*
    *(Renders: Inverter DC-AC Efficiency Curve Line Chart)*
*   *"How well does the plant output correlate with irradiance? Are there clipping or soiling events?"*
    *(Renders: Irradiance vs Power Output Scatter Chart)*
*   *"Are there any thermal anomalies or hotspot risks in the module temperature data?"*
    *(Renders: Module Thermal Profile Trend Chart)*
*   *"What is the plant availability and how many grid curtailment events have occurred?"*
    *(Renders: Operating State Breakdown Doughnut Chart)*

### 2. Specific Asset Telemetry
Ask about any of the 42 assets. The agent resolves the name, retrieves the latest database values, and plots trends:

*   *"Show me insights on MFM 12"* or *"active power of B2MFM12"*
    *(Renders: Active Power Trend Line Chart for B2MFM12)*
*   *"What is the health and cycle history of battery B1BCT1?"*
    *(Renders: Battery SoH Doughnut Chart)*
*   *"What are the latest readings for meteorological station PPCWMS1?"*
    *(Renders: Ambient vs Module Temp Line Chart)*
*   *"Retrieve telemetry for converter B2DCCON1"*
    *(Renders: Input Voltage Trend Line Chart)*
*   *"Active power of PCS B1PCS1"*
    *(Renders: Active Power Trend Line Chart)*

### 3. Asset Comparisons
Compare multiple units. The agent automatically extracts the relevant comparison metric and displays a comparison bar chart:

*   *"Compare B2_BCT3 and B2_BCT1"*
    *(Renders: Battery SOC / SoH Comparison Bar Chart)*
*   *"Compare meters B1MFM1 and B3MFM2"*
    *(Renders: Active Power Comparison Bar Chart)*
*   *"Compare all inverters"* or *"compare B1INV1 and B3INV1"*
    *(Renders: Conversion Efficiency Comparison Bar Chart)*
*   *"Compare B2DCCON1, B2DCCON2, and B2DCCON3"*
    *(Renders: Telemetry Voltage/Power Comparison Bar Chart)*

### 4. Administrative & Inventory Listings
List active assets by type:

*   *"List down all assets by type"*
    *(Displays: Complete, grouped list of all 42 assets with database keys)*
*   *"Show all meters"* or *"list available battery assets"*
    *(Displays: Grouped active assets for the requested category)*

### 5. Direct SQL Access
Run raw SQL queries against the database directly:

*   *"SELECT timestamp, ambientTemperature, planeOfArraySensor01 FROM telemetry WHERE device_group='WEATHER' ORDER BY timestamp DESC LIMIT 5"*
*   *"SELECT count(*), device_group FROM telemetry GROUP BY device_group"*

### 6. Methodology & Capabilities Queries
Ask the agent how any of the diagnostics work or what it can do:

*   *"What can you do?"* or *"What are your capabilities?"*
*   *"How do you calculate PR gap attribution?"*
*   *"How does string outlier detection work?"*
*   *"How do you calibrate the soiling rate?"*
*   *"How is battery SoH calculated?"*
*   *"How does the inverter efficiency curve work?"*
*   *"How do you detect power clipping?"*
*   *"How does NOCT thermal hotspot detection work?"*
*   *"How is plant availability calculated?"*

---

## 📚 ML Diagnostics & Calculations Methodology (How It Works)

Here are the precise mathematical formulas and algorithmic steps used by the agent to perform each diagnostic:

### 1. Performance Ratio (PR) & Gap Attribution
*   **Expected Solar Power ($P_{\text{exp}}$)**:
    $$P_{\text{exp}} = \text{Capacity} \cdot \left(\frac{\text{POA}}{1000}\right) \cdot (1 - 0.004 \cdot (T_{\text{mod}} - 25)) \cdot \text{LossFactor}$$
    *Where $\text{Capacity} = 8648\text{ kW}$, $T_{\text{mod}}$ is module temperature, $\text{POA}$ is Plane-of-Array irradiance, and $\text{LossFactor} = 0.85$.*
*   **Energy Integration**: Calculated in 5-minute intervals ($\Delta t = \frac{5}{60}\text{ hours}$) where $E = P \cdot \Delta t$.
*   **PR Actual**: $PR = \frac{E_{\text{actual}}}{E_{\text{expected}}}$ (represented as a decimal).
*   **Gap Decomposition**:
    *   *Grid Curtailment*: Lost energy during periods where `inverterStatus = 2` (curtailed).
    *   *Hardware Inefficiency*: Losses when DC-AC efficiency drops below 92%.
    *   *Soiling & Shading*: Distributed from residual energy losses using season-calibrated shares.

### 2. String Combiner Box (SCB) Current Outliers
*   **Statistical Outliers**: For the selected inverter, retrieves the latest SCB current readings ($x_i$). Computes mean ($\mu$) and standard deviation ($\sigma$).
*   **Z-Score Calculation**:
    $$Z_i = \frac{x_i - \mu}{\sigma}$$
*   **Outlier Flag**: Strings with $Z_i < -2.0$ are flagged as underperforming (e.g., due to localized shading or module faults).

### 3. Unsupervised Soiling Loss Calibration
*   **Wash Cycle Detection**: Looks for sudden daily PR jumps ($PR_{\text{today}} - PR_{\text{yesterday}} > 4.0\%$) to identify panel washing events.
*   **Decline Rate**: Performs a linear regression on daily PR values between consecutive cleaning events. The negative slope ($\%/ \text{day}$) represents the daily dust accumulation/soiling rate.

### 4. Battery (BESS) Health & Cycle Tracker
*   **Coulombic Efficiency ($\eta_C$)**:
    $$\eta_C = \frac{\sum |I_{\text{discharge}}|}{\sum |I_{\text{charge}}|}$$
*   **Cycle Counting**: A full cycle is equivalent to a cumulative SOC change of 200%:
    $$\text{Cycles} = \left\lfloor \frac{\sum |\Delta SOC|}{200} \right\rfloor$$
*   **State of Health (SoH)**: Estimated from cycle wear using a linear fade coefficient of 0.015% per cycle:
    $$SoH = 100\% - (0.015\% \cdot \text{Cycles})$$

### 5. Inverter DC-AC Efficiency Curve
*   **Efficiency**: $\text{Efficiency} = \frac{P_{\text{AC}}}{P_{\text{DC}}} \cdot 100\%$ (where $P_{\text{AC}}$ is `outputPower` and $P_{\text{DC}}$ is `inputPVPower`).
*   **Load Factor Binning**: Bins inverter operating data into 10% load increments relative to the $1430\text{ kW}$ inverter rating.
*   **Degradation Alert**: Flags an inverter as underperforming if its average conversion efficiency falls below $92\%$.

### 6. Irradiance-Power Correlation & Clipping
*   **Correlation ($R^2$)**: Calculates the coefficient of determination ($R^2$) of the linear regression between WMS POA irradiance and aggregate active power.
*   **Clipping Detection**: Flags clipping events when plant active power stays within 2% of the aggregate rated capacity ($8648\text{ kW}$) while POA irradiance exceeds $800\text{ W/m}^2$.

### 7. Module Thermal Hotspots (NOCT Model)
*   **Predicted Module Temp ($T_{\text{pred}}$)**:
    $$T_{\text{pred}} = T_{\text{ambient}} + \left(\frac{\text{NOCT} - 20}{800}\right) \cdot \text{POA}$$
    *Where $\text{NOCT} = 45^\circ\text{C}$ (Nominal Operating Cell Temperature).*
*   **Hotspot Criteria**: Flags module anomalies where measured module temperature sensor exceeds predicted temperature by more than $8^\circ\text{C}$ ($T_{\text{measured}} - T_{\text{pred}} > 8^\circ\text{C}$).

### 8. Daylight Availability & Curtailment
*   **Daytime Filter**: Restricts active state classifications (Running, Standby, Grid Curtailment, Scheduled Maintenance, Fault/Trip) strictly to daylight hours where $\text{POA} > 50\text{ W/m}^2$.
*   **Daylight Availability**:
    $$\text{Availability} = \frac{\text{Running Hours}}{\text{Total Daylight Hours}} \cdot 100\%$$

