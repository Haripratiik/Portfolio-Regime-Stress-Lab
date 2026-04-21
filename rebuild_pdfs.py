#!/usr/bin/env python3
"""Rebuild the report PDF (2 pages) and slide PDF (1 page) from pre-saved
figures and metrics.json.

Run from the submission folder root:
    python rebuild_pdfs.py
"""
import json, textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
FIGS    = BASE / 'figures'
DATA    = BASE / 'data'
OUT_RPT = BASE / 'portfolio_regime_stress_lab_report.pdf'
OUT_SLD = BASE / 'portfolio_regime_stress_lab_slide.pdf'

metrics = json.loads((DATA / 'metrics.json').read_text())
rs      = metrics['regimeSummary']
fm      = metrics['forecast']
ti      = metrics['topInsights']
sym     = ', '.join(metrics['symbols'])
ORDER   = tuple(fm['modelOrder'])
ORDER_S = f'({ORDER[0]},{ORDER[1]},{ORDER[2]})'

REGIME_LABELS = ['Calm Expansion', 'Rotation / Transition', 'Stress Contagion']
REGIME_COLORS = {
    'Calm Expansion':        '#2ea043',
    'Rotation / Transition': '#d29922',
    'Stress Contagion':      '#f85149',
}
ROW_BG = ['#f0fdf4', '#fefce8', '#fff1f2']

# ── Helpers ────────────────────────────────────────────────────────────────────
def _img(ax, name):
    path = FIGS / f'{name}.png'
    if path.exists():
        ax.imshow(mpimg.imread(str(path)))
    ax.axis('off')


def _rule(fig, y, x0=0.03, x1=0.97, color='#cbd5e1'):
    fig.add_artist(Line2D([x0, x1], [y, y],
                          transform=fig.transFigure, color=color, lw=0.8))


def _section(fig, y, label):
    """Section header + rule. Returns y of the rule."""
    fig.text(0.03, y + 0.013, label, fontsize=9, fontweight='bold', color='#1e3a5f')
    _rule(fig, y + 0.008)
    return y + 0.008


def _hdr_bar(fig, label, y0=0.93, height=0.07):
    ax = fig.add_axes([0.0, y0, 1.0, height])
    ax.set_facecolor('#1e3a5f')
    ax.axis('off')
    fig.text(0.03, y0 + height * 0.65, label,
             fontsize=15, fontweight='bold', color='white', va='center')
    return ax


# ── Report ─────────────────────────────────────────────────────────────────────
def build_report():
    with PdfPages(str(OUT_RPT)) as pdf:

        # ── PAGE 1: written summary ───────────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')

        _hdr_bar(fig, 'Portfolio Regime & Stress Propagation Lab')
        fig.text(0.03, 0.954, 'Quant Mentorship Final Project',
                 fontsize=9, color='#93c5fd', va='center')

        # DATASET + METHODS (two-column)
        _section(fig, 0.886, 'DATASET')
        dataset = [
            ('Universe',  sym),
            ('Period',    f'{metrics["startDate"]}  to  {metrics["endDate"]}'),
            ('Frequency', 'Daily adjusted-close OHLCV'),
            ('Portfolio', 'Equal-weight synthetic portfolio'),
            ('Source',    'Yahoo Finance via yfinance (pre-downloaded in data/)'),
        ]
        for i, (k, v) in enumerate(dataset):
            yy = 0.875 - i * 0.030
            fig.text(0.03, yy, k + ':',  fontsize=8.5, color='#6b7280', fontweight='bold')
            fig.text(0.14, yy, v, fontsize=8.5, color='#111827')

        fig.text(0.52, 0.897, 'METHODS', fontsize=9, fontweight='bold', color='#1e3a5f')
        _rule(fig, 0.894, x0=0.52)
        methods = [
            '5 rolling state features: realized vol, avg pairwise corr,',
            '  breadth above 50-DMA, cross-sectional dispersion, lead-lag conc.',
            'Stress Propagation Score: equal-weight z-score blend of 5 metrics',
            'Regime detection: K-Means (k=3, n_init=25, random_state=42)',
            'Forecasting: ARIMA(2,0,1) on 20-day realized volatility',
            'Risk: GBM Monte Carlo (500 paths, 252-day forward horizon)',
        ]
        for i, line in enumerate(methods):
            fig.text(0.52, 0.875 - i * 0.030, line, fontsize=8.5, color='#111827')

        # REGIME SUMMARY TABLE
        _section(fig, 0.716, 'REGIME SUMMARY')
        col_x    = [0.03, 0.23, 0.34, 0.46, 0.59, 0.72, 0.84]
        col_hdrs = ['Regime', 'Days', '% of Sample', 'Avg Vol (20d)', 'Avg Corr', 'Avg Breadth', 'Stress Score']
        for x, hh in zip(col_x, col_hdrs):
            fig.text(x, 0.702, hh, fontsize=7.5, fontweight='bold', color='#374151')
        for ri, label in enumerate(REGIME_LABELS):
            d  = rs[label]
            ry = 0.690 - ri * 0.034
            fig.add_artist(Rectangle((0.02, ry - 0.008), 0.96, 0.030,
                                     transform=fig.transFigure,
                                     facecolor=ROW_BG[ri], edgecolor='none', zorder=0))
            fig.add_artist(Rectangle((0.02, ry - 0.008), 0.007, 0.030,
                                     transform=fig.transFigure,
                                     facecolor=REGIME_COLORS[label], edgecolor='none', zorder=1))
            vals = [
                label, str(d['count']),
                f'{d["pctOfSample"]:.1f}%',
                f'{d["avgRealizedVol20d"]:.1%}',
                f'{d["avgCorr20d"]:.2f}',
                f'{d["avgBreadth"]:.2f}',
                f'{d["avgStressScore"]:+.2f}',
            ]
            for x, val in zip(col_x, vals):
                fig.text(x + 0.01, ry + 0.004, val, fontsize=8.5, color='#111827', zorder=2)

        # KEY FINDINGS
        _section(fig, 0.572, 'KEY FINDINGS')
        for i, finding in enumerate(ti):
            fy = 0.553 - i * 0.046
            fig.add_artist(Circle((0.046, fy + 0.012), 0.004,
                                  transform=fig.transFigure, color='#1e3a5f', zorder=2))
            fig.text(0.060, fy, textwrap.fill(finding, width=95),
                     fontsize=10, color='#111827', linespacing=1.4)

        # ARIMA METRICS BAR
        bar_y = 0.212
        fig.add_artist(Rectangle((0.02, bar_y), 0.96, 0.072,
                                  transform=fig.transFigure,
                                  facecolor='#eff6ff', edgecolor='#bfdbfe', lw=0.8, zorder=0))
        fig.text(0.035, bar_y + 0.052, 'ARIMA MODEL METRICS',
                 fontsize=9, fontweight='bold', color='#1e3a5f', zorder=1)
        fig.text(0.035, bar_y + 0.025,
                 f'Model: ARIMA{ORDER_S}     '
                 f'RMSE: {fm["rmse"]:.4f}  (naive: {fm["baselineRmse"]:.4f})     '
                 f'MAE: {fm["mae"]:.4f}',
                 fontsize=9.5, color='#1e40af', zorder=1)
        fig.text(0.035, bar_y + 0.004,
                 f'AIC: {fm["aic"]:.2f}     BIC: {fm["bic"]:.2f}     '
                 f'Walk-forward evaluation over {252}+ test steps',
                 fontsize=9.5, color='#1e40af', zorder=1)

        # BRIEF REFLECTION
        refl_y = 0.130
        _rule(fig, refl_y + 0.02)
        reflection = (
            'The core finding is that market regimes are economically distinct and statistically '
            'separable. Stress Contagion days carry 6x higher realized volatility than Calm '
            'Expansion, with sharply elevated cross-stock correlation and depressed market '
            'breadth. ARIMA(2,0,1) confirms that volatility -- unlike returns -- is forecastable, '
            'supporting volatility-based position sizing in the Portfolio Manager.'
        )
        fig.text(0.03, refl_y + 0.005, textwrap.fill(reflection, width=115),
                 fontsize=9, color='#374151', linespacing=1.45)

        # Footer
        _rule(fig, 0.030, color='#e5e7eb')
        fig.text(0.03, 0.012,
                 f'Universe: {sym}  |  {metrics["startDate"]} to {metrics["endDate"]}  |  Data: Yahoo Finance (yfinance)',
                 fontsize=7.5, color='#9ca3af')
        fig.text(0.97, 0.012, '1 / 2', fontsize=7.5, color='#9ca3af', ha='right')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 2: all six charts ────────────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')

        _hdr_bar(fig, 'Portfolio Regime & Stress Propagation Lab  —  Visual Summary',
                 y0=0.95, height=0.05)

        # 3-row x 2-col grid. Each chart: width=0.45, height=0.277
        # Columns at x=0.03 and x=0.52; rows at y=0.645, 0.340, 0.040
        charts = [
            ('stress_regimes',    'Stress Propagation Score with Regime Shading'),
            ('forecast_actual',   'ARIMA(2,0,1): Forecast vs Actual Realized Vol'),
            ('normalized_prices', 'Normalized Price Paths (Base = 1.0 on day 1)'),
            ('rolling_state',     'Rolling Portfolio State Metrics'),
            ('transition_heatmap','Regime Transition Probability Heatmap'),
            ('lead_lag_heatmap',  'Lead-Lag Heatmap (Leader on Y, Follower on X)'),
        ]
        CW = 0.45
        CH = 0.277
        CL = [0.03, 0.52]
        CB = [0.648, 0.346, 0.044]  # bottom y per row (top row first)

        for idx, (name, title) in enumerate(charts):
            col = idx % 2
            row = idx // 2
            ax = fig.add_axes([CL[col], CB[row], CW, CH])
            _img(ax, name)
            ax.set_title(title, fontsize=8, pad=3, color='#374151')

        _rule(fig, 0.030, color='#e5e7eb')
        fig.text(0.97, 0.012, '2 / 2', fontsize=7.5, color='#9ca3af', ha='right')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f'Report written -> {OUT_RPT.name}')


# ── Slide ──────────────────────────────────────────────────────────────────────
def build_slide():
    with PdfPages(str(OUT_SLD)) as pdf:
        fig = plt.figure(figsize=(13.333, 7.5))
        fig.patch.set_facecolor('#0d1117')

        # LEFT: hero chart (stress regimes) ────────────────────────────────────
        ax = fig.add_axes([0.02, 0.07, 0.51, 0.87])
        _img(ax, 'stress_regimes')
        ax.set_title('Stress Propagation Score & Regime Shading',
                     fontsize=9, color='#6b7280', pad=4)

        # Separator
        fig.add_artist(Line2D([0.553, 0.553], [0.06, 0.97],
                              transform=fig.transFigure, color='#21262d', lw=1.5))

        # RIGHT: content ────────────────────────────────────────────────────────
        RX = 0.568

        # Title
        fig.text(RX, 0.950, 'Stress & Calm Regimes Are Sharply Distinct',
                 fontsize=16, fontweight='bold', color='#f0f6fc', va='top')
        fig.text(RX, 0.900,
                 f'{sym}  |  2021-2026  |  Equal-weight portfolio',
                 fontsize=9, color='#6b7280', va='top')

        # Three metric cards
        cards = [
            ('7.65%',  'Calm avg vol\n(20d realized)',  '#0c1a3a', '#60a5fa'),
            ('45.90%', 'Stress avg vol\n(20d realized)', '#3b0a0a', '#f87171'),
            ('0.0269', f'ARIMA{ORDER_S}\nRMSE',          '#0c1a3a', '#60a5fa'),
        ]
        CW_C, CH_C, GAP = 0.125, 0.108, 0.010
        CY = 0.765
        for i, (val, lbl, bg, fg) in enumerate(cards):
            cx = RX + i * (CW_C + GAP)
            fig.add_artist(Rectangle((cx, CY), CW_C, CH_C,
                                     transform=fig.transFigure,
                                     facecolor=bg, edgecolor=fg, lw=1.5, zorder=0))
            fig.text(cx + CW_C / 2, CY + CH_C * 0.65, val,
                     fontsize=14, fontweight='bold', color=fg,
                     ha='center', va='center', transform=fig.transFigure, zorder=1)
            fig.text(cx + CW_C / 2, CY + CH_C * 0.22, lbl,
                     fontsize=7, color='#94a3b8',
                     ha='center', va='center', transform=fig.transFigure, zorder=1)

        # Three findings (bulleted)
        findings = [
            ('Volatility', 'Stress Contagion averaged 45.90% annualized realized vol\n'
                           'vs 7.65% in Calm Expansion -- a 6x difference.'),
            ('Regime Score', 'Stress Propagation Score averaged +1.49 in Stress\n'
                             'vs -0.93 in Calm -- economically distinct states.'),
            ('Forecast', f'ARIMA{ORDER_S} RMSE {fm["rmse"]:.4f} outperforms naive baseline\n'
                         f'at {fm["baselineRmse"]:.4f} across walk-forward evaluation.'),
        ]
        fy = 0.720
        for label, text in findings:
            fig.add_artist(Rectangle((RX, fy - 0.005), 0.060, 0.025,
                                     transform=fig.transFigure,
                                     facecolor='#172554', edgecolor='none', zorder=0))
            fig.text(RX + 0.030, fy + 0.005, label,
                     fontsize=7.5, fontweight='bold', color='#93c5fd',
                     ha='center', va='center', transform=fig.transFigure, zorder=1)
            fig.text(RX + 0.068, fy, text,
                     fontsize=9.5, color='#e2e8f0', va='top', linespacing=1.45)
            fy -= 0.110

        # Footer bar
        FY = 0.040
        fig.add_artist(Rectangle((0.555, FY), 0.440, 0.058,
                                  transform=fig.transFigure,
                                  facecolor='#161b22', edgecolor='#21262d', lw=1, zorder=0))
        fig.text(RX, FY + 0.038,
                 f'K-Means (k=3, n_init=25)  |  ARIMA{ORDER_S}  |  '
                 f'GBM Monte Carlo (500 paths, 252-day horizon)',
                 fontsize=8, color='#6b7280', zorder=1)
        fig.text(RX, FY + 0.012,
                 f'AIC: {fm["aic"]:.1f}  |  BIC: {fm["bic"]:.1f}  |  '
                 f'MAE: {fm["mae"]:.4f}  |  Naive RMSE: {fm["baselineRmse"]:.4f}',
                 fontsize=8, color='#6b7280', zorder=1)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f'Slide  written -> {OUT_SLD.name}')


if __name__ == '__main__':
    build_report()
    build_slide()
