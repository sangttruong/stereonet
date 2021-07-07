import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser, Namespace
import os
import numpy as np
from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
def add_args(parser: ArgumentParser):
    # General arguments
    parser.add_argument('--prediction_data_path', type=str, help='Path the the folder that contains the csv files that contains targets and predictions. Inside this folder, the two csv files are required: preds_on_val.csv and preds_on_train.csv')
    
    parser.add_argument('--output_path', type=str, help='Directory to save the residual plot to')


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).
    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    return args

   
def plot_diagnostics(residuals, variable=0, lags=40, fig=None, figsize=(15,7), savefig = False, path = None):
  
  _import_mpl()
  fig = create_mpl_fig(fig, figsize)

  # # Eliminate residuals associated with burned or diffuse likelihoods
  # d = np.maximum(self.loglikelihood_burn, self.nobs_diffuse)
  # resid = self.filter_results.standardized_forecasts_error[variable, d:]
  # loglikelihood_burn: the number of observations during which the likelihood is not evaluated.

  # Standardize residual
  # Source: https://alkaline-ml.com/pmdarima/1.1.1/_modules/pmdarima/arima/arima.html
  resid = residuals
  resid = (resid - np.nanmean(resid)) / np.nanstd(resid)

  # Top-left: residuals vs time
  ax = fig.add_subplot(221)
#  if hasattr(self.data, 'dates') and self.data.dates is not None:
#      x = self.data.dates[d:]._mpl_repr()
#  else:
#      x = np.arange(len(resid))
  x = np.arange(len(resid))
  ax.plot(x, resid)
  ax.hlines(0, x[0], x[-1], alpha=0.5)
  ax.set_xlim(x[0], x[-1])
  ax.set_title('Standardized residual')

  # Top-right: histogram, Gaussian kernel density, Normal density
  # Can only do histogram and Gaussian kernel density on the non-null
  # elements
  resid_nonmissing = resid[~(np.isnan(resid))]
  ax = fig.add_subplot(222)

  # gh5792: Remove  except after support for matplotlib>2.1 required
  try:
      ax.hist(resid_nonmissing, density=True, label='Hist')
  except AttributeError:
      ax.hist(resid_nonmissing, normed=True, label='Hist')

  from scipy.stats import gaussian_kde, norm
  kde = gaussian_kde(resid_nonmissing)
  xlim = (-1.96*2, 1.96*2)
  x = np.linspace(xlim[0], xlim[1])
  ax.plot(x, kde(x), label='KDE')
  ax.plot(x, norm.pdf(x), label='N(0,1)')
  ax.set_xlim(xlim)
  ax.legend()
  ax.set_title('Histogram plus estimated density')

  # Bottom-left: QQ plot
  ax = fig.add_subplot(223)
  from statsmodels.graphics.gofplots import qqplot
  qqplot(resid_nonmissing, line='s', ax=ax)
  ax.set_title('Normal Q-Q')

  # Bottom-right: Correlogram
  ax = fig.add_subplot(224)
  from statsmodels.graphics.tsaplots import plot_pacf
  plot_pacf(resid, ax=ax, lags=lags)
  ax.set_title('Partial Autocorrelation function')
 
  ax.set_ylim(-0.1, 0.1)

  if savefig == True:
    fig.suptitle('Residual diagnostic', fontsize = 20)
    fig.savefig(path, dpi = 500)
    fig.show()
  return fig

def residual_plot(args, filename, plotname):
    # val predictions
    csv_path = os.path.join(args.prediction_data_path, filename)
    df = pd.read_csv(csv_path)
    residual = np.array(df['label']) - np.array(df['prediction'])
    plot_diagnostics(residual, savefig=True, path=os.path.join(args.prediction_data_path, plotname))
#    plt.hist(residual, bins=20)
#    plt.savefig(os.path.join(args.prediction_data_path, plotname))
    
 
    
if __name__ == "__main__":
    args = parse_train_args()
    # val predictions
    residual_plot(args, 'preds_on_val.csv', 'val_residual.png')
    # train predictions
    residual_plot(args, 'preds_on_train.csv', 'train_residual.png')
    
    
    
