import sys, sqlite3, pandas as pd
DB = r'C:\Users\saxen\.gemini\antigravity\brain\0f95910d-307c-40cb-b421-02dc23fbd684\scratch\sungenie_telemetry.db'
conn = sqlite3.connect(DB)
w = pd.read_sql_query("SELECT DISTINCT timestamp FROM telemetry WHERE device_group='WEATHER' LIMIT 5", conn)
i = pd.read_sql_query("SELECT DISTINCT timestamp FROM telemetry WHERE device_group='INVERTER' LIMIT 5", conn)
print('WEATHER ts:', w['timestamp'].tolist())
print('INVERTER ts:', i['timestamp'].tolist())
wset = set(pd.read_sql_query("SELECT DISTINCT timestamp FROM telemetry WHERE device_group='WEATHER'", conn)['timestamp'])
iset = set(pd.read_sql_query("SELECT DISTINCT timestamp FROM telemetry WHERE device_group='INVERTER'", conn)['timestamp'])
print(f'W count: {len(wset)} | I count: {len(iset)} | Overlap: {len(wset & iset)}')

# Also check inverterStatus distribution
inv_stat = pd.read_sql_query("SELECT inverterStatus, COUNT(*) as cnt FROM telemetry WHERE device_group='INVERTER' GROUP BY inverterStatus", conn)
print('InverterStatus distribution:')
print(inv_stat)

# Check inputPVPower sample
pv = pd.read_sql_query("SELECT inputPVPower, outputPower FROM telemetry WHERE device_group='INVERTER' AND inputPVPower IS NOT NULL LIMIT 5", conn)
print('PV power sample:', pv)
conn.close()
