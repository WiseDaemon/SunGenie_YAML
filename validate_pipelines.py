import sys
sys.path.insert(0, r'C:\Users\saxen\.gemini\antigravity\brain\0f95910d-307c-40cb-b421-02dc23fbd684\scratch')
import importlib
import ml_pipelines
importlib.reload(ml_pipelines)

print("=== All ML Pipeline Checks ===")

r = ml_pipelines.get_expected_vs_actual_generation()
print("[1] PR Gap: actual=%s expected=%s PR=%s" % (r["actual_kwh"], r["expected_kwh"], r["pr_actual"]))

r = ml_pipelines.detect_scb_outliers()
print("[2] SCB: mean=%s outliers=%d" % (r.get("mean_current"), len(r.get("underperforming_strings", {}))))

r = ml_pipelines.calibrate_soiling_rate()
print("[3] Soiling: rate=%s%%/day cleanings=%d" % (r["avg_daily_soiling_rate_pct"], len(r["inferred_cleaning_dates"])))

r = ml_pipelines.get_bess_health()
print("[4] BESS: SoH=%s%% cycles=%s" % (r["state_of_health_pct"], r["total_cycles"]))

r = ml_pipelines.analyze_inverter_efficiency()
print("[5] Efficiency: fleet_avg=%s%% underperf=%s" % (r["fleet_avg_efficiency_pct"], r["underperforming_inverters"]))

r = ml_pipelines.analyze_irradiance_power_correlation()
print("[6] Irradiance: R2=%s ratio=%s flag=%s pts=%d" % (r["correlation_r2"], r["avg_output_ratio"], r["anomaly_flag"], len(r["scatter_data"])))

r = ml_pipelines.detect_thermal_anomalies()
print("[7] Thermal: anomalies=%s max_delta=%sC status=%s" % (r["anomaly_count"], r["max_delta_c"], r["status"]))

r = ml_pipelines.analyze_grid_curtailment()
print("[8] Curtailment: avail=%s%% faults=%sh breakdown=%s" % (r["plant_availability_pct"], r["total_fault_hours"], list(r["category_breakdown_hours"].keys())))

print("\nALL PIPELINES OK")
