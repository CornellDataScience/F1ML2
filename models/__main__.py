import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

import F1_models
import pickle


if __name__ == "__main__":
    f1 = F1_models.F1_regressions(data_file='data/HOLY_after_race3.csv')
    f1.run_all_models()
    print(f1.preds)
    print(f1.comparison_dict)
    f1.save_linear('race3_linear.pickle')
    pickle.dump(f1.linear_scaler, open('linear_scaler.pkl', 'wb'))
