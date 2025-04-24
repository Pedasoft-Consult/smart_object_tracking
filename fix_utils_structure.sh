#!/bin/bash

# Navigate to the project root
cd /mnt/c/job_projects/smart_object_tracking || {
  echo "❌ Project directory not found!"
  exit 1
}

echo "📁 Creating utils folder if it doesn't exist..."
mkdir -p utils

echo "🚚 Moving utility files into utils/..."
mv connectivity.py utils/ 2>/dev/null
mv dataloaders.py utils/ 2>/dev/null
mv try_except.py utils/ 2>/dev/null
mv __init__.py utils/ 2>/dev/null

echo "🧹 Cleaning up Python bytecode cache..."
find . -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "✅ Structure fixed. Your utils package is now set up."

echo "🚀 You can now run:"
echo "python main.py --source car-detection.mp4 --tracker byte_track --display"
