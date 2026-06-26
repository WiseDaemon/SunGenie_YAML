import sqlite3
import os

DB_PATH = r"C:\Users\saxen\.gemini\antigravity\brain\0f95910d-307c-40cb-b421-02dc23fbd684\scratch\sungenie_telemetry.db"

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get distinct groups
    cursor.execute("SELECT DISTINCT device_group FROM telemetry")
    groups = [r[0] for r in cursor.fetchall()]
    
    print(f"Device Groups: {groups}")
    
    # For each group, print non-null columns and a sample row
    for group in groups:
        print(f"\n==========================================")
        print(f"Group: {group}")
        print(f"==========================================")
        
        # Get count
        cursor.execute("SELECT COUNT(*) FROM telemetry WHERE device_group = ?", (group,))
        count = cursor.fetchone()[0]
        print(f"Total records: {count}")
        
        # Get one sample row
        cursor.execute(f"SELECT * FROM telemetry WHERE device_group = ? LIMIT 1", (group,))
        sample = cursor.fetchone()
        if not sample:
            print("No records found.")
            continue
            
        cols = [col[0] for col in cursor.description]
        
        # Find non-null columns for this group
        non_null_cols = []
        for col in cols:
            cursor.execute(f"SELECT COUNT(*) FROM telemetry WHERE device_group = ? AND {col} IS NOT NULL", (group,))
            non_null_count = cursor.fetchone()[0]
            if non_null_count > 0:
                non_null_cols.append((col, non_null_count))
                
        print("Non-null columns (and record count):")
        for col, cnt in non_null_cols:
            print(f"  - {col}: {cnt}")
            
        print("\nSample values:")
        sample_dict = dict(zip(cols, sample))
        for col, cnt in non_null_cols:
            print(f"  - {col}: {sample_dict[col]}")
            
    conn.close()

if __name__ == "__main__":
    main()
