# scan_ble.py
import asyncio
from bleak import BleakScanner

async def scan(secs=5):
    print("Escaneando BLE durante", secs, "s...")
    devices = await BleakScanner.discover(timeout=secs)
    for d in devices:
        print(d)

if __name__ == "__main__":
    asyncio.run(scan(6))
