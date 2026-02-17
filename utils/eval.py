import numpy as np
import pandas as pd
from deepxde.metrics import mean_squared_error, l2_relative_error


def eval_predictions(real, pred, compartiments=["S", "I", "beta"]):

    indexes = range(len(compartiments))

    return pd.DataFrame({
        "compartiment": compartiments, 
        "RMSE": [np.sqrt(mean_squared_error(real[:,i], pred[:,i])) for i in indexes],
        "L2": [l2_relative_error(real[:,i], pred[:,i]) for i in indexes],
        "L-infinity": [np.max(np.abs(real[:,i] - pred[:,i])) for i in indexes]
    })


def format_latex_table(df):
    return df.to_latex()
    
    # latex_code = df.to_latex(
    #     index=False,
    #     escape=False,
    #     column_format='cccc',
    #     header=['\\textbf{compartiment}', '\\textbf{RMSE}', '$\\mathbf{\\mathcal{L}_2}$', '$\\mathbf{\\mathcal{L}_\\infty}$'],
    #     caption='Valores das métricas de erro (\\textit{RMSE}, norma $\\mathcal{L}_2$ e norma $\\mathcal{L}_\\infty$) para as soluções aproximadas pela rede neural, em comparação com as soluções analíticas.',
    #     label='tab:metricas-sen-beta-noisy-dfe',
    #     position='H'
    # )

    # latex_code = latex_code.replace('β', '$\\beta$')
    # latex_code = latex_code.replace('S', '$S$')
    # latex_code = latex_code.replace('I', '$I$')
    # latex_code = latex_code.replace('tabular', 'tabular{cccc}') 
    # latex_code = latex_code.replace('\\toprule', '\\hline')
    # latex_code = latex_code.replace('\\midrule', '')
    # latex_code = latex_code.replace('\\bottomrule', '\\hline')

    # final_latex = """\\begin{table}[H]
    # \\centering
    # """ + latex_code.split('\\begin{tabular}', 1)[-1]

    # return final_latex