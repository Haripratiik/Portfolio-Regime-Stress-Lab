#!/usr/bin/env python3
"""Rebuild report (3 pages) and slide (1 page) PDFs.
Run from the submission folder: python rebuild_pdfs.py
"""
import json, textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ── Paths & data ───────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
FIGS    = BASE / 'figures'
DATA    = BASE / 'data'
OUT_RPT = BASE / 'portfolio_regime_stress_lab_report.pdf'
OUT_SLD = BASE / 'portfolio_regime_stress_lab_slide.pdf'

M       = json.loads((DATA / 'metrics.json').read_text())
rs      = M['regimeSummary']
fm      = M['forecast']
ti      = M['topInsights']
sym     = ', '.join(M['symbols'])
ORDER_S = f'({fm["modelOrder"][0]},{fm["modelOrder"][1]},{fm["modelOrder"][2]})'

RLABELS = ['Calm Expansion', 'Rotation / Transition', 'Stress Contagion']
RCOLORS = {
    'Calm Expansion':        '#2ea043',
    'Rotation / Transition': '#d29922',
    'Stress Contagion':      '#f85149',
}
ROW_BG  = ['#f0fdf4', '#fefce8', '#fff1f2']
HDR_BG  = '#1e3a5f'
BODY    = '#111827'
SUB     = '#6b7280'
RULE_C  = '#cbd5e1'

# ── Shared helpers ─────────────────────────────────────────────────────────────
def _img(ax, name):
    path = FIGS / f'{name}.png'
    if path.exists():
        ax.imshow(mpimg.imread(str(path)))
    ax.axis('off')

def _rule(fig, y, x0=0.03, x1=0.97, c=RULE_C):
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=c, lw=0.8))

def _section(fig, y, label):
    fig.text(0.03, y + 0.013, label, fontsize=9, fontweight='bold', color=HDR_BG)
    _rule(fig, y + 0.008)

def _hdr_bar(fig, title, sub=None, y0=0.93, h=0.07):
    ax = fig.add_axes([0.0, y0, 1.0, h])
    ax.set_facecolor(HDR_BG); ax.axis('off')
    fig.text(0.03, y0 + h * 0.65, title,
             fontsize=15, fontweight='bold', color='white', va='center')
    if sub:
        fig.text(0.03, y0 + h * 0.22, sub, fontsize=9, color='#93c5fd', va='center')

def _footer(fig, page, total):
    _rule(fig, 0.034, c='#e5e7eb')
    fig.text(0.03, 0.015,
             f'Universe: {sym}  |  {M["startDate"]} to {M["endDate"]}  |  Yahoo Finance via yfinance',
             fontsize=7.5, color='#9ca3af')
    fig.text(0.97, 0.015, f'{page} / {total}', fontsize=7.5, color='#9ca3af', ha='right')


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT  (3 pages)
# ══════════════════════════════════════════════════════════════════════════════
def build_report():
    with PdfPages(str(OUT_RPT)) as pdf:

        # ── PAGE 1: Introduction, Dataset, Methods ─────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        _hdr_bar(fig, 'Portfolio Regime & Stress Propagation Lab',
                 sub='Quant Mentorship Final Project')

        _section(fig, 0.876, 'STUDY OVERVIEW')
        overview = (
            'This report investigates how market stress builds and propagates across an equal-weight '
            'portfolio of six mega-cap technology stocks: AAPL, MSFT, GOOGL, AMZN, META, and NVDA. '
            'Five years of daily OHLCV data (April 2021 to April 2026) were sourced from Yahoo '
            'Finance, cleaned to a 1,305-trading-day aligned price matrix, and analyzed using '
            'rolling time-series features, unsupervised clustering, and statistical forecasting. '
            'The central research question is whether a custom Stress Propagation Score -- built from '
            'five daily state metrics -- can meaningfully separate calm, transitional, and stressed '
            'market environments, and whether those environments carry materially different risk '
            'profiles for portfolio managers.'
        )
        fig.text(0.03, 0.823, textwrap.fill(overview, width=112),
                 fontsize=9.5, color=BODY, linespacing=1.55)

        # Dataset column
        _section(fig, 0.694, 'DATASET')
        data_rows = [
            ('Universe',      sym),
            ('Period',        f'{M["startDate"]}  to  {M["endDate"]}'),
            ('Frequency',     'Daily adjusted-close OHLCV'),
            ('Portfolio',     'Equal-weight synthetic portfolio'),
            ('Source',        'Yahoo Finance via yfinance (pre-downloaded; no API call required)'),
            ('Trading days',  '1,305 after temporal alignment and forward-fill cleaning'),
            ('Preprocessing', 'Forward-fill at most 2 consecutive gaps (exchange holiday mismatches);'
                              ' then drop rows with any remaining NaN'),
        ]
        for i, (k, v) in enumerate(data_rows):
            yy = 0.675 - i * 0.033
            fig.text(0.03, yy, k + ':', fontsize=8.5, color=SUB, fontweight='bold')
            fig.text(0.185, yy, v, fontsize=8.5, color=BODY)

        # Methods column
        _section(fig, 0.694, '')   # right col, no label
        fig.text(0.52, 0.707, 'ANALYTICAL METHODS', fontsize=9, fontweight='bold', color=HDR_BG)
        _rule(fig, 0.702, x0=0.52)
        methods = [
            ('Rolling features',    '5 state metrics: realized vol, avg pairwise corr,\n'
                                    '  breadth above 50-DMA, dispersion, lead-lag conc.'),
            ('Stress Score',        'Equal-weight z-score blend of the 5 rolling metrics;\n'
                                    '  also included as 6th K-Means clustering feature'),
            ('Regime detection',    'K-Means (k=3, n_init=25, random_state=42)\n'
                                    '  on 6-feature rolling state matrix'),
            ('Stationarity',        'Augmented Dickey-Fuller (ADF) on prices,\n'
                                    '  log returns, realized vol, and stress score'),
            ('Autocorrelation',     'Ljung-Box on portfolio log returns (market\n'
                                    '  efficiency check) and on ARIMA residuals'),
            (f'ARIMA{ORDER_S}',     '20-day realized vol forecast; AIC/BIC model\n'
                                    '  selection; expanding walk-forward evaluation'),
            ('GBM Monte Carlo',     '500 paths, 252-day forward horizon;\n'
                                    '  VaR (95/99%) and CVaR (95/99%) computed'),
        ]
        for i, (k, v) in enumerate(methods):
            yy = 0.675 - i * 0.042
            fig.text(0.52, yy, k + ':', fontsize=8.5, color=SUB, fontweight='bold')
            fig.text(0.68,  yy, v, fontsize=8.5, color=BODY, linespacing=1.3)

        _footer(fig, 1, 3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 2: Regime Analysis, ARIMA, Key Findings ──────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        _hdr_bar(fig, 'Portfolio Regime & Stress Propagation Lab',
                 sub='Regime Analysis & Key Findings')

        # Regime table via matplotlib table (auto-sizes, no overflow)
        _section(fig, 0.864, 'MARKET REGIME SUMMARY')

        tbl_data = []
        for lbl in RLABELS:
            d = rs[lbl]
            tbl_data.append([
                lbl,
                str(d['count']),
                f'{d["pctOfSample"]:.1f}%',
                f'{d["avgRealizedVol20d"]:.1%}',
                f'{d["avgCorr20d"]:.2f}',
                f'{d["avgBreadth"]:.2f}',
                f'{d["avgStressScore"]:+.2f}',
            ])
        col_hdrs = ['Regime', 'Days', '% of\nSample', 'Avg Vol\n(20d ann.)',
                    'Avg Corr\n(20d)', 'Avg\nBreadth', 'Avg Stress\nScore']
        col_w    = [0.28, 0.07, 0.09, 0.12, 0.10, 0.09, 0.11]

        tbl_ax = fig.add_axes([0.02, 0.718, 0.96, 0.140])
        tbl_ax.axis('off')
        tbl = tbl_ax.table(cellText=tbl_data, colLabels=col_hdrs,
                            colWidths=col_w, cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 2.1)

        for j in range(len(col_hdrs)):
            c = tbl[(0, j)]
            c.set_facecolor(HDR_BG)
            c.set_text_props(color='white', fontweight='bold', fontsize=8)
        for ri, lbl in enumerate(RLABELS):
            for j in range(len(col_hdrs)):
                c = tbl[(ri + 1, j)]
                c.set_facecolor(ROW_BG[ri])
                if j == 0:
                    c.set_text_props(color=RCOLORS[lbl], fontweight='bold', fontsize=8.5)

        # Regime interpretation paragraphs
        _section(fig, 0.696, 'REGIME INTERPRETATION')
        regime_paras = [
            ('Calm Expansion',
             f'Vol: {rs["Calm Expansion"]["avgRealizedVol20d"]:.1%}  |  '
             f'Corr: {rs["Calm Expansion"]["avgCorr20d"]:.2f}  |  '
             f'Breadth: {rs["Calm Expansion"]["avgBreadth"]:.2f}',
             'Low volatility and high breadth characterize an orderly uptrend. Cross-stock '
             'correlation is low (0.14), meaning stocks are largely moving on their own '
             'fundamentals rather than in lock-step. Regime persistence is 99.8% per day -- '
             'once calm, the portfolio tends to remain calm for extended periods.'),
            ('Rotation / Transition',
             f'Vol: {rs["Rotation / Transition"]["avgRealizedVol20d"]:.1%}  |  '
             f'Corr: {rs["Rotation / Transition"]["avgCorr20d"]:.2f}  |  '
             f'Breadth: {rs["Rotation / Transition"]["avgBreadth"]:.2f}',
             'Intermediate stress with rising correlation (0.37) and declining breadth (0.54). '
             'This regime captures sector rotation, modest pullbacks, and macro uncertainty '
             'that has not yet triggered broad contagion. It serves as the leading indicator '
             'for Stress Contagion entries, with a 2.8% daily probability of escalating.'),
            ('Stress Contagion',
             f'Vol: {rs["Stress Contagion"]["avgRealizedVol20d"]:.1%}  |  '
             f'Corr: {rs["Stress Contagion"]["avgCorr20d"]:.2f}  |  '
             f'Breadth: {rs["Stress Contagion"]["avgBreadth"]:.2f}',
             'Sharply elevated volatility (45.90%) with correlation spiking to 0.82 -- stocks '
             'move in near lock-step, eliminating the diversification benefit of holding six '
             'names. Breadth collapses to 0.14, meaning fewer than 15% of stocks are above '
             'their 50-day moving average. Once entered, stress persists 95.2% of trading days.'),
        ]
        ry = 0.670
        for key, stats_line, body in regime_paras:
            col = RCOLORS[key]
            fig.add_artist(Rectangle((0.03, ry - 0.004), 0.006, 0.024,
                                     transform=fig.transFigure,
                                     facecolor=col, edgecolor='none', zorder=1))
            fig.text(0.043, ry + 0.006, key, fontsize=9, fontweight='bold', color=col, zorder=2)
            fig.text(0.043, ry - 0.010, stats_line, fontsize=8, color=SUB)
            fig.text(0.043, ry - 0.030, textwrap.fill(body, width=103),
                     fontsize=8.5, color=BODY, linespacing=1.35)
            ry -= 0.110

        # ARIMA section
        _section(fig, 0.335, 'ARIMA VOLATILITY FORECASTING')
        arima_txt = (
            f'Realized volatility was confirmed stationary by ADF (p < 0.001), making it an '
            f'appropriate ARIMA target. An AIC/BIC grid search over five candidate orders '
            f'(AR=1-2, I=0-1, MA=0-1) selected ARIMA{ORDER_S}. Walk-forward evaluation over '
            f'250+ expanding-window steps produced RMSE {fm["rmse"]:.4f} and MAE {fm["mae"]:.4f}, '
            f'beating the lag-1 naive baseline (RMSE {fm["baselineRmse"]:.4f}, '
            f'MAE {fm["baselineMae"]:.4f}). Ljung-Box tests on residuals confirm white-noise '
            f'at lags 10 and 20 (p > 0.05), validating the model specification. '
            f'Model fit: AIC = {fm["aic"]:.1f},  BIC = {fm["bic"]:.1f}.'
        )
        fig.text(0.03, 0.285, textwrap.fill(arima_txt, width=112),
                 fontsize=9.5, color=BODY, linespacing=1.55)

        # Key findings
        _section(fig, 0.215, 'KEY FINDINGS')
        for i, finding in enumerate(ti):
            fy = 0.200 - i * 0.036
            fig.add_artist(Rectangle((0.03, fy - 0.001), 0.005, 0.020,
                                     transform=fig.transFigure,
                                     facecolor=HDR_BG, edgecolor='none', zorder=1))
            fig.text(0.043, fy + 0.002, finding, fontsize=9, color=BODY, zorder=2)

        _footer(fig, 2, 3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── PAGE 3: Visual Summary ─────────────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        _hdr_bar(fig, 'Portfolio Regime & Stress Propagation Lab  —  Visual Summary',
                 y0=0.952, h=0.048)

        charts = [
            ('stress_regimes',    'Stress Propagation Score with Regime Shading'),
            ('forecast_actual',   f'ARIMA{ORDER_S}: Walk-Forward Forecast vs Actual Vol'),
            ('normalized_prices', 'Normalized Price Paths (Base = 1.0 on first day)'),
            ('rolling_state',     'Rolling Portfolio State Metrics (20d window)'),
            ('transition_heatmap','Regime Transition Probability Heatmap'),
            ('lead_lag_heatmap',  'Lead-Lag Heatmap (Leader on Y, Follower on X)'),
        ]
        CW, CH = 0.455, 0.272
        CL     = [0.025, 0.520]
        CB     = [0.648, 0.350, 0.054]

        for idx, (name, title) in enumerate(charts):
            ax = fig.add_axes([CL[idx % 2], CB[idx // 2], CW, CH])
            _img(ax, name)
            ax.set_title(title, fontsize=8, pad=3, color='#374151')

        _footer(fig, 3, 3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f'Report -> {OUT_RPT.name}')


# ══════════════════════════════════════════════════════════════════════════════
#  SLIDE  (1 page, 16:9)
# ══════════════════════════════════════════════════════════════════════════════
def build_slide():
    with PdfPages(str(OUT_SLD)) as pdf:
        fig = plt.figure(figsize=(13.333, 7.5))
        fig.patch.set_facecolor('#0d1117')

        # White left panel so the chart PNG (white bg) looks seamless
        fig.add_artist(Rectangle((0.0, 0.0), 0.545, 1.0,
                                  transform=fig.transFigure,
                                  facecolor='#f8fafc', edgecolor='none', zorder=0))
        # Thin accent bar at top of left panel
        fig.add_artist(Rectangle((0.0, 0.955), 0.545, 0.045,
                                  transform=fig.transFigure,
                                  facecolor=HDR_BG, edgecolor='none', zorder=1))
        fig.text(0.015, 0.977, 'Portfolio Regime & Stress Propagation Lab',
                 fontsize=9, color='#93c5fd', va='center', zorder=2)

        # Chart (fills white panel cleanly)
        ax = fig.add_axes([0.015, 0.085, 0.520, 0.855])
        ax.set_facecolor('#f8fafc')
        _img(ax, 'stress_regimes')

        # Caption under chart
        fig.text(0.272, 0.052, 'Fig. 1 — Stress Propagation Score with regime shading (green = Calm, amber = Rotation, red = Stress)',
                 fontsize=7, color=SUB, ha='center', zorder=2)

        # Vertical divider
        fig.add_artist(Line2D([0.548, 0.548], [0.0, 1.0],
                              transform=fig.transFigure, color='#e2e8f0', lw=1.5))

        # ── RIGHT PANEL ────────────────────────────────────────────────────────
        RX = 0.562

        # Title
        fig.text(RX, 0.950, 'Stress & Calm Regimes\nAre Sharply Distinct',
                 fontsize=17, fontweight='bold', color='#f0f6fc', va='top', linespacing=1.25)

        # Context (universe + period + what the chart shows)
        fig.text(RX, 0.875, f'{sym}   |   2021 – 2026   |   Equal-weight portfolio',
                 fontsize=8.5, color='#6b7280', va='top')
        context = (
            'K-Means clustering on six rolling state features separates 1,305 trading '
            'days into three distinct regimes. Calm and Stress are economically and '
            'statistically separable: volatility is 6x higher in Stress, cross-stock '
            'correlation rises from 0.14 to 0.82, and market breadth collapses from '
            '0.63 to 0.14 -- conditions that materially elevate portfolio risk.'
        )
        fig.text(RX, 0.835, textwrap.fill(context, width=53),
                 fontsize=8.5, color='#cbd5e1', va='top', linespacing=1.48)

        # Three metric cards
        cards = [
            ('7.65%',  'Calm avg\nrealized vol',  '#0c1a3a', '#60a5fa'),
            ('45.90%', 'Stress avg\nrealized vol', '#3b0a0a', '#f87171'),
            ('×6.0',   'Volatility\nmultiple',     '#1a1f0f', '#a3e635'),
        ]
        CW_C, CH_C, GAP_C = 0.120, 0.095, 0.009
        CY_C = 0.618
        for i, (val, lbl, bg, fg) in enumerate(cards):
            cx = RX + i * (CW_C + GAP_C)
            fig.add_artist(Rectangle((cx, CY_C), CW_C, CH_C,
                                     transform=fig.transFigure,
                                     facecolor=bg, edgecolor=fg, lw=1.5, zorder=1))
            fig.text(cx + CW_C / 2, CY_C + CH_C * 0.65, val,
                     fontsize=14, fontweight='bold', color=fg,
                     ha='center', va='center', transform=fig.transFigure, zorder=2)
            fig.text(cx + CW_C / 2, CY_C + CH_C * 0.22, lbl,
                     fontsize=7, color='#94a3b8',
                     ha='center', va='center', transform=fig.transFigure, zorder=2)

        # Three findings with label + two-line explanation
        findings = [
            ('Regime Separation',
             f'Stress Contagion averaged 45.90% annualized vol vs 7.65% in Calm Expansion.',
             'Correlation rose from 0.14 to 0.82 -- diversification benefit largely disappears.'),
            ('Stress Propagation Score',
             f'Custom score (+1.49 in Stress, -0.93 in Calm) blends vol, corr, breadth,',
             'dispersion, and lead-lag concentration into one interpretable state variable.'),
            (f'ARIMA{ORDER_S} Forecast',
             f'Walk-forward RMSE {fm["rmse"]:.4f} beats naive baseline at {fm["baselineRmse"]:.4f}.',
             'Residuals are white noise (Ljung-Box p > 0.05), confirming model adequacy.'),
        ]
        fy = 0.578
        for label, line1, line2 in findings:
            # Label row
            fig.add_artist(Rectangle((RX, fy - 0.002), 0.425, 0.030,
                                     transform=fig.transFigure,
                                     facecolor='#1e293b', edgecolor='none', zorder=1))
            fig.text(RX + 0.008, fy + 0.010, label,
                     fontsize=8.5, fontweight='bold', color='#60a5fa', va='center', zorder=2)
            # Explanation lines
            fig.text(RX + 0.008, fy - 0.018, line1, fontsize=8.5, color='#e2e8f0', va='top')
            fig.text(RX + 0.008, fy - 0.038, line2, fontsize=8.5, color='#94a3b8', va='top')
            fy -= 0.098

        # Footer
        FY = 0.035
        fig.add_artist(Rectangle((0.548, FY), 0.452, 0.048,
                                  transform=fig.transFigure,
                                  facecolor='#161b22', edgecolor='#21262d', lw=0.8, zorder=1))
        fig.text(RX, FY + 0.030,
                 f'K-Means (k=3, n_init=25)  |  ARIMA{ORDER_S}  |  ADF + Ljung-Box diagnostics  |  GBM Monte Carlo (500 paths, 252d)',
                 fontsize=7.5, color='#6b7280', zorder=2)
        fig.text(RX, FY + 0.010,
                 f'AIC {fm["aic"]:.1f}  |  BIC {fm["bic"]:.1f}  |  '
                 f'RMSE {fm["rmse"]:.4f}  |  MAE {fm["mae"]:.4f}  |  '
                 f'Naive RMSE {fm["baselineRmse"]:.4f}',
                 fontsize=7.5, color='#6b7280', zorder=2)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f'Slide  -> {OUT_SLD.name}')


if __name__ == '__main__':
    build_report()
    build_slide()
