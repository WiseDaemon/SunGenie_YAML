import sqlite3
import os

DB_PATH = r"C:\Users\saxen\.gemini\antigravity\brain\0f95910d-307c-40cb-b421-02dc23fbd684\scratch\sungenie_telemetry.db"

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Creating alerts table...")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        asset_id TEXT,
        severity TEXT,
        message TEXT,
        status TEXT
    )
    """)

    print("Creating indices for performance optimization...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry (timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_device_group ON telemetry (device_group)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_meter_id ON telemetry (meterId)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_group_timestamp ON telemetry (device_group, timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_meter_timestamp ON telemetry (meterId, timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp)")

    # Insert a few mock historical alerts to start with
    cursor.execute("SELECT COUNT(*) FROM alerts")
    if cursor.fetchone()[0] == 0:
        print("Inserting sample alerts...")
        sample_alerts = [
            ("2026-06-20 10:15:00", "JAMNAGAR_VIRTUAL_GATEWAY_B1INV1", "Warning", "Inverter B1INV1 efficiency dropped below 92%", "Resolved"),
            ("2026-06-22 14:30:00", "JAMNAGAR_VIRTUAL_GATEWAY_PPCWMS1", "Warning", "High wind speed alert - module structure safety check recommended", "Resolved"),
            ("2026-06-24 11:20:00", "JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1", "Warning", "BESS Cell voltage imbalance warning", "Active")
        ]
        cursor.executemany("INSERT INTO alerts (timestamp, asset_id, severity, message, status) VALUES (?, ?, ?, ?, ?)", sample_alerts)

    conn.commit()
    conn.close()
    print("Database updates completed successfully.")

if __name__ == "__main__":
    main()
