import os
import glob
import csv
import json
import sqlite3
from datetime import datetime

DATA_DIR = r"C:\LLM\SunGenie data"
DB_PATH = r"C:\Users\saxen\.gemini\antigravity\brain\0f95910d-307c-40cb-b421-02dc23fbd684\scratch\sungenie_telemetry.db"

def determine_device_group(meter_id):
    if not meter_id:
        return 'OTHER'
    parts = meter_id.split('_')
    suffix = parts[-1]
    if 'INV' in suffix:
        return 'INVERTER'
    elif 'MFM' in suffix:
        return 'METER'
    elif 'BCT' in suffix:
        return 'BESS'
    elif 'PCS' in suffix:
        return 'PCS'
    elif 'WMS' in suffix or 'WEATHER' in suffix or 'WS' in suffix:
        return 'WEATHER'
    elif 'PQM' in suffix:
        return 'PQM'
    elif 'DCCON' in suffix:
        return 'DCCON'
    return suffix

def parse_timestamp(ts_str):
    if not ts_str:
        return None
    try:
        ts_str = ts_str.strip()
        dt = datetime.strptime(ts_str, "%Y-%m-%d %I:%M:%S %p")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return ts_str

def main():
    print("Initializing SQLite Database...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Removed existing database.")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        type TEXT,
        meterId TEXT,
        deviceUID TEXT,
        device_group TEXT,
        activePower REAL,
        energyToday REAL,
        energyTillDate REAL,
        apparentEnergyTillDate REAL,
        apparentEnergyToday REAL,
        inputPVPower REAL,
        outputPower REAL,
        inputCurrent REAL,
        inputVoltage REAL,
        inverterStatus INTEGER,
        bessSOC REAL,
        bessCurrent REAL,
        voltageRPhase REAL,
        currentRPhase REAL,
        totalActiveEnergyImport REAL,
        totalActiveEnergyExport REAL,
        netActiveEnergy REAL,
        windSpeed REAL,
        windDirection REAL,
        ambientTemperature REAL,
        humidity REAL,
        globalHorizontalIrradiance REAL,
        planeOfArraySensor01 REAL,
        moduleTemperatureSensor01 REAL,
        moduleTemperatureSensor03 REAL,
        rainfall REAL,
        scb_currents TEXT
    )
    """)
    
    conn.commit()
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process.")
    
    total_records = 0
    batch = []
    
    float_cols = [
        'activePower', 'energyToday', 'energyTillDate', 'apparentEnergyTillDate', 'apparentEnergyToday',
        'inputPVPower', 'outputPower', 'inputCurrent', 'inputVoltage', 'bessSOC', 'bessCurrent', 'voltageRPhase',
        'currentRPhase', 'totalActiveEnergyImport', 'totalActiveEnergyExport', 'netActiveEnergy',
        'windSpeed', 'windDirection', 'ambientTemperature', 'humidity', 'globalHorizontalIrradiance',
        'planeOfArraySensor01', 'moduleTemperatureSensor01', 'moduleTemperatureSensor03', 'rainfall'
    ]
    
    for idx, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/{len(csv_files)}] Processing {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                continue
            
            headers = [h.strip() for h in headers]
            header_map = {h: idx for idx, h in enumerate(headers) if h}
            
            for row in reader:
                if not row:
                    continue
                
                type_idx = header_map.get('type')
                rec_type = row[type_idx] if type_idx is not None and type_idx < len(row) else 'UNKNOWN'
                
                time_idx = header_map.get('time') or header_map.get('tms')
                raw_time = row[time_idx] if time_idx is not None and time_idx < len(row) else None
                formatted_time = parse_timestamp(raw_time)
                
                meter_idx = header_map.get('meterId')
                meter_id = row[meter_idx] if meter_idx is not None and meter_idx < len(row) else None
                
                device_uid_idx = header_map.get('deviceUID')
                device_uid = row[device_uid_idx] if device_uid_idx is not None and device_uid_idx < len(row) else None
                
                device_group = determine_device_group(meter_id)
                
                values = {
                    'timestamp': formatted_time,
                    'type': rec_type,
                    'meterId': meter_id,
                    'deviceUID': device_uid,
                    'device_group': device_group,
                    'inverterStatus': None,
                    'scb_currents': None
                }
                
                inv_status_idx = header_map.get('inverterStatus')
                if inv_status_idx is not None and inv_status_idx < len(row) and row[inv_status_idx]:
                    try:
                        values['inverterStatus'] = int(float(row[inv_status_idx]))
                    except ValueError:
                        pass
                
                for col in float_cols:
                    col_idx = header_map.get(col)
                    val = None
                    if col_idx is not None and col_idx < len(row) and row[col_idx]:
                        val_str = row[col_idx].strip()
                        if val_str and val_str != '-3276.800048828125' and val_str != '-3276.8':
                            try:
                                val = float(val_str)
                            except ValueError:
                                pass
                    values[col] = val
                
                if device_group == 'INVERTER':
                    scb_dict = {}
                    for i in range(1, 25):
                        scb_col = f"SCB{i}current"
                        scb_idx = header_map.get(scb_col)
                        if scb_idx is not None and scb_idx < len(row) and row[scb_idx]:
                            val_str = row[scb_idx].strip()
                            if val_str and val_str != '-3276.800048828125' and val_str != '-3276.8':
                                try:
                                    scb_dict[f"SCB{i}"] = float(val_str)
                                except ValueError:
                                    pass
                    if scb_dict:
                        values['scb_currents'] = json.dumps(scb_dict)
                
                row_tuple = (
                    values['timestamp'], values['type'], values['meterId'], values['deviceUID'], values['device_group'],
                    values['activePower'], values['energyToday'], values['energyTillDate'], values['apparentEnergyTillDate'],
                    values['apparentEnergyToday'], values['inputPVPower'], values['outputPower'], values['inputCurrent'], values['inputVoltage'],
                    values['inverterStatus'], values['bessSOC'], values['bessCurrent'], values['voltageRPhase'],
                    values['currentRPhase'], values['totalActiveEnergyImport'], values['totalActiveEnergyExport'],
                    values['netActiveEnergy'], values['windSpeed'], values['windDirection'], values['ambientTemperature'],
                    values['humidity'], values['globalHorizontalIrradiance'], values['planeOfArraySensor01'],
                    values['moduleTemperatureSensor01'], values['moduleTemperatureSensor03'], values['rainfall'],
                    values['scb_currents']
                )
                
                batch.append(row_tuple)
                total_records += 1
                
                if len(batch) >= 10000:
                    cursor.executemany("""
                    INSERT INTO telemetry (
                        timestamp, type, meterId, deviceUID, device_group,
                        activePower, energyToday, energyTillDate, apparentEnergyTillDate,
                        apparentEnergyToday, inputPVPower, outputPower, inputCurrent, inputVoltage,
                        inverterStatus, bessSOC, bessCurrent, voltageRPhase,
                        currentRPhase, totalActiveEnergyImport, totalActiveEnergyExport,
                        netActiveEnergy, windSpeed, windDirection, ambientTemperature,
                        humidity, globalHorizontalIrradiance, planeOfArraySensor01,
                        moduleTemperatureSensor01, moduleTemperatureSensor03, rainfall,
                        scb_currents
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, batch)
                    conn.commit()
                    batch = []
                    
    if batch:
        cursor.executemany("""
        INSERT INTO telemetry (
            timestamp, type, meterId, deviceUID, device_group,
            activePower, energyToday, energyTillDate, apparentEnergyTillDate,
            apparentEnergyToday, inputPVPower, outputPower, inputCurrent, inputVoltage,
            inverterStatus, bessSOC, bessCurrent, voltageRPhase,
            currentRPhase, totalActiveEnergyImport, totalActiveEnergyExport,
            netActiveEnergy, windSpeed, windDirection, ambientTemperature,
            humidity, globalHorizontalIrradiance, planeOfArraySensor01,
            moduleTemperatureSensor01, moduleTemperatureSensor03, rainfall,
            scb_currents
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, batch)
        conn.commit()
        
    print(f"Total rows inserted: {total_records}")
    
    print("Creating database indexes for query performance...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry(timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_meter ON telemetry(meterId);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_group ON telemetry(device_group);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_type ON telemetry(type);")
    conn.commit()
    
    conn.close()
    print("Database compilation complete!")

if __name__ == "__main__":
    main()
