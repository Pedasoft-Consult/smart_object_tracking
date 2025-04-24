# Run this script as Administrator in PowerShell

# Step 1: List available USB devices
Write-Host "📋 Available USB Devices for WSL:"
$devices = usbipd wsl list | Where-Object { $_ -match '^\s*\d' }

if (-not $devices) {
    Write-Host "⚠️ No USB devices found or usbipd is not installed." -ForegroundColor Yellow
    exit
}

$devices | ForEach-Object { Write-Host $_ }

# Step 2: Ask user to select a device
$deviceNumber = Read-Host "`n🔢 Enter the number of the device you want to attach (e.g., 1)"

# Step 3: Extract BusID from the selected device
$selectedLine = $devices | Where-Object { $_ -match "^\s*$deviceNumber\s+" }
if (-not $selectedLine) {
    Write-Host "❌ Invalid selection." -ForegroundColor Red
    exit
}

$busid = ($selectedLine -split '\s+')[1]

# Step 4: Attach the device
Write-Host "`n🔌 Attaching device with BusID: $busid ..."
usbipd wsl attach --busid $busid

Write-Host "`n✅ Done. Check your WSL instance for the device."
