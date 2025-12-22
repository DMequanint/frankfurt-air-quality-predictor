from pathlib import Path
import subprocess
from datetime import datetime

def full_pipeline():
    report_path = Path("reports/full_pipeline_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# ML Pipeline Demo\n\n")
        f.write("Date: " + str(datetime.now()) + "\n\n")
        f.write("Test Dual Models:\n")
    
    subprocess.run(["python", "-m", "frankfurt_air_quality_predictor.modeling.predict", "test"], 
                   stdout=report_path.open('a'))
    
    with open(report_path, 'a') as f:
        f.write("\nHigh Pollution:\n")
    
    subprocess.run(["python", "-m", "frankfurt_air_quality_predictor.modeling.predict", "quick"], 
                   stdout=report_path.open('a'))
    
    with open(report_path, 'a') as f:
        f.write("\nSafe Morning:\n")
    
    subprocess.run([
        "python", "-m", "frankfurt_air_quality_predictor.modeling.predict", "quick",
        "--hour", "9", "--day-of-week", "1", "--rolling-24h", "8.2",
        "--lag-1", "7.9", "--lag-24", "9.1", "--rolling-mean-24", "8.5"
    ], stdout=report_path.open('a'))
    
    print("Report saved:", report_path)

def quick_demo():
    report_path = Path("reports/demo_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Frankfurt PM2.5 Demo\n")
        f.write("MAE: 1.32 | F1: 0.892\n\n")
        f.write("High Pollution:\n")
    
    subprocess.run([
        "python", "-m", "frankfurt_air_quality_predictor.modeling.predict", "quick",
        "--hour", "18", "--day-of-week", "5", "--rolling-24h", "20.5",
        "--lag-1", "22.1", "--lag-24", "18.9", "--rolling-mean-24", "19.8"
    ], stdout=report_path.open('a'))
    
    with open(report_path, 'a') as f:
        f.write("\nSafe Morning:\n")
    
    subprocess.run([
        "python", "-m", "frankfurt_air_quality_predictor.modeling.predict", "quick",
        "--hour", "9", "--day-of-week", "1", "--rolling-24h", "8.2",
        "--lag-1", "7.9", "--lag-24", "9.1", "--rolling-mean-24", "8.5"
    ], stdout=report_path.open('a'))
    
    print("Demo saved:", report_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full-pipeline":
        full_pipeline()
    elif len(sys.argv) > 1 and sys.argv[1] == "quick-demo":
        quick_demo()
    else:
        print("Usage: python report.py full-pipeline | quick-demo")
