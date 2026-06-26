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
