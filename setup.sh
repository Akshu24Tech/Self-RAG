#!/bin/bash

echo "🚀 Universal Self-RAG Setup"
echo "======================================"

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create .env if not exists
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your GOOGLE_API_KEY"
else
    echo "✅ .env file already exists"
fi

# Create documents folder
mkdir -p documents

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Get your FREE Gemini API key: https://makersuite.google.com/app/apikey"
echo "2. Edit .env and add your API key"
echo "3. Add PDF files to documents/ folder"
echo "4. Run: python example.py"
echo ""
echo "======================================"
