import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import logging

def debug_evidently():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=df, current_data=df)
    print("Return type:", type(result))
    print("Attributes:", dir(result))

if __name__ == "__main__":
    debug_evidently()
