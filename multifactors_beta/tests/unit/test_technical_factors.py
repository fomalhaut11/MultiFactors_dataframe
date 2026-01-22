"""
Technical Factors 单元测试

测试MomentumCalculator, ReversalCalculator, VolumeCalculator
以及TechnicalIndicators的新增方法
"""

import pytest
import numpy as np
import pandas as pd


def create_mock_price_data(n_dates=100, n_stocks=5):
    """创建模拟价格数据"""
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    stocks = [f'Stock_{i}' for i in range(n_stocks)]

    # 创建MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, stocks],
        names=['TradingDates', 'StockCodes']
    )

    # 生成随机价格数据
    np.random.seed(42)

    # 基础价格
    base_prices = np.random.uniform(10, 100, n_stocks)

    # 生成价格序列（带有随机游走）
    prices = []
    for stock_base in base_prices:
        stock_prices = [stock_base]
        for _ in range(n_dates - 1):
            change = np.random.normal(0, 0.02)
            stock_prices.append(stock_prices[-1] * (1 + change))
        prices.extend(stock_prices)

    # 重新排列为按日期排序
    prices_array = np.array(prices).reshape(n_stocks, n_dates).T.flatten()

    return pd.Series(prices_array, index=index, name='close')


def create_mock_ohlcv_data(n_dates=100, n_stocks=5):
    """创建模拟OHLCV数据"""
    close = create_mock_price_data(n_dates, n_stocks)

    # 生成high, low, volume
    np.random.seed(43)
    high = close * (1 + np.abs(np.random.normal(0, 0.02, len(close))))
    low = close * (1 - np.abs(np.random.normal(0, 0.02, len(close))))
    volume = pd.Series(
        np.random.uniform(1e6, 1e8, len(close)),
        index=close.index,
        name='volume'
    )

    return {
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


class TestMomentumCalculator:
    """MomentumCalculator测试类"""

    @pytest.fixture
    def close_data(self):
        """创建收盘价数据"""
        return create_mock_price_data(n_dates=300, n_stocks=3)

    def test_calculate_momentum(self, close_data):
        """测试基本动量计算"""
        from factors.generators.technical import MomentumCalculator

        momentum = MomentumCalculator.calculate_momentum(close_data, period=20)

        # 验证输出格式
        assert isinstance(momentum, pd.Series)
        assert momentum.index.equals(close_data.index)

        # 前period个值应该是NaN（因为pct_change需要回看）
        for stock in close_data.index.get_level_values('StockCodes').unique():
            stock_mom = momentum.xs(stock, level='StockCodes')
            assert stock_mom.iloc[:20].isna().all()

    def test_calculate_momentum_1m(self, close_data):
        """测试1个月动量"""
        from factors.generators.technical import MomentumCalculator

        mom_1m = MomentumCalculator.calculate_momentum_1m(close_data)

        assert isinstance(mom_1m, pd.Series)
        # 验证非全NaN
        assert not mom_1m.isna().all()

    def test_calculate_momentum_12m_skip1m(self, close_data):
        """测试12个月动量（跳过1个月）"""
        from factors.generators.technical import MomentumCalculator

        mom_12m_skip = MomentumCalculator.calculate_momentum_12m_skip1m(close_data)

        assert isinstance(mom_12m_skip, pd.Series)

    def test_calculate_risk_adjusted_momentum(self, close_data):
        """测试风险调整动量"""
        from factors.generators.technical import MomentumCalculator

        risk_adj_mom = MomentumCalculator.calculate_risk_adjusted_momentum(
            close_data, period=60
        )

        assert isinstance(risk_adj_mom, pd.Series)

    def test_calculate_momentum_consistency(self, close_data):
        """测试动量一致性"""
        from factors.generators.technical import MomentumCalculator

        consistency = MomentumCalculator.calculate_momentum_consistency(
            close_data, period=63, sub_period=21
        )

        assert isinstance(consistency, pd.Series)
        # 一致性应在0-1之间
        valid_values = consistency.dropna()
        assert (valid_values >= 0).all() and (valid_values <= 1).all()


class TestReversalCalculator:
    """ReversalCalculator测试类"""

    @pytest.fixture
    def price_data(self):
        """创建价格数据"""
        return create_mock_ohlcv_data(n_dates=100, n_stocks=3)

    def test_calculate_short_term_reversal(self, price_data):
        """测试短期反转"""
        from factors.generators.technical import ReversalCalculator

        reversal = ReversalCalculator.calculate_short_term_reversal(
            price_data['close'], period=5
        )

        assert isinstance(reversal, pd.Series)
        # 验证非全NaN
        assert not reversal.isna().all()

    def test_calculate_reversal_1w(self, price_data):
        """测试1周反转"""
        from factors.generators.technical import ReversalCalculator

        reversal_1w = ReversalCalculator.calculate_reversal_1w(price_data['close'])

        assert isinstance(reversal_1w, pd.Series)

    def test_calculate_volume_reversal(self, price_data):
        """测试成交量反转"""
        from factors.generators.technical import ReversalCalculator

        vol_reversal = ReversalCalculator.calculate_volume_reversal(
            price_data['close'],
            price_data['volume'],
            return_period=5,
            volume_period=20
        )

        assert isinstance(vol_reversal, pd.Series)

    def test_calculate_max_drawdown_reversal(self, price_data):
        """测试最大回撤反转"""
        from factors.generators.technical import ReversalCalculator

        dd_reversal = ReversalCalculator.calculate_max_drawdown_reversal(
            price_data['close'], period=21
        )

        assert isinstance(dd_reversal, pd.Series)
        # 回撤值应为正（因为取了负号）
        valid_values = dd_reversal.dropna()
        assert (valid_values >= 0).all()


class TestVolumeCalculator:
    """VolumeCalculator测试类"""

    @pytest.fixture
    def price_data(self):
        """创建OHLCV数据"""
        return create_mock_ohlcv_data(n_dates=100, n_stocks=3)

    def test_calculate_vwap(self, price_data):
        """测试VWAP计算"""
        from factors.generators.technical import VolumeCalculator

        vwap = VolumeCalculator.calculate_vwap(
            price_data['high'],
            price_data['low'],
            price_data['close'],
            price_data['volume'],
            period=20
        )

        assert isinstance(vwap, pd.Series)
        # VWAP应该有有效值
        assert not vwap.isna().all()
        # VWAP应该为正值
        valid_values = vwap.dropna()
        assert (valid_values > 0).all()

    def test_calculate_obv(self, price_data):
        """测试OBV计算"""
        from factors.generators.technical import VolumeCalculator

        obv = VolumeCalculator.calculate_obv(
            price_data['close'],
            price_data['volume']
        )

        assert isinstance(obv, pd.Series)

    def test_calculate_mfi(self, price_data):
        """测试MFI计算"""
        from factors.generators.technical import VolumeCalculator

        mfi = VolumeCalculator.calculate_mfi(
            price_data['high'],
            price_data['low'],
            price_data['close'],
            price_data['volume'],
            period=14
        )

        assert isinstance(mfi, pd.Series)
        # MFI应在0-100之间
        valid_values = mfi.dropna()
        assert (valid_values >= 0).all() and (valid_values <= 100).all()

    def test_calculate_volume_ratio(self, price_data):
        """测试成交量比率"""
        from factors.generators.technical import VolumeCalculator

        vol_ratio = VolumeCalculator.calculate_volume_ratio(
            price_data['volume'], period=20
        )

        assert isinstance(vol_ratio, pd.Series)
        # 比率应为正
        valid_values = vol_ratio.dropna()
        assert (valid_values > 0).all()

    def test_calculate_accumulation_distribution(self, price_data):
        """测试A/D指标"""
        from factors.generators.technical import VolumeCalculator

        ad = VolumeCalculator.calculate_accumulation_distribution(
            price_data['high'],
            price_data['low'],
            price_data['close'],
            price_data['volume']
        )

        assert isinstance(ad, pd.Series)


class TestTechnicalIndicatorsExtended:
    """TechnicalIndicators扩展方法测试"""

    @pytest.fixture
    def price_data(self):
        """创建OHLCV数据（简单Series格式用于单股票测试）"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='D')

        base_price = 100
        prices = [base_price]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))

        close = pd.Series(prices, index=dates)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))

        return {'high': high, 'low': low, 'close': close}

    def test_kdj(self, price_data):
        """测试KDJ指标"""
        from factors.generators.technical import TechnicalIndicators

        k, d, j = TechnicalIndicators.kdj(
            price_data['high'],
            price_data['low'],
            price_data['close'],
            n=9, m1=3, m2=3
        )

        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        assert isinstance(j, pd.Series)

        # K和D应在0-100范围
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_cci(self, price_data):
        """测试CCI指标"""
        from factors.generators.technical import TechnicalIndicators

        cci = TechnicalIndicators.cci(
            price_data['high'],
            price_data['low'],
            price_data['close'],
            window=20
        )

        assert isinstance(cci, pd.Series)

    def test_williams_r(self, price_data):
        """测试Williams %R指标"""
        from factors.generators.technical import TechnicalIndicators

        williams = TechnicalIndicators.williams_r(
            price_data['high'],
            price_data['low'],
            price_data['close'],
            window=14
        )

        assert isinstance(williams, pd.Series)
        # Williams %R应在-100到0之间
        valid_values = williams.dropna()
        assert (valid_values >= -100).all() and (valid_values <= 0).all()

    def test_atr(self, price_data):
        """测试ATR指标"""
        from factors.generators.technical import TechnicalIndicators

        atr = TechnicalIndicators.atr(
            price_data['high'],
            price_data['low'],
            price_data['close'],
            window=14
        )

        assert isinstance(atr, pd.Series)
        # ATR应为正
        valid_values = atr.dropna()
        assert (valid_values >= 0).all()

    def test_dmi(self, price_data):
        """测试DMI指标"""
        from factors.generators.technical import TechnicalIndicators

        plus_di, minus_di, adx = TechnicalIndicators.dmi(
            price_data['high'],
            price_data['low'],
            price_data['close'],
            window=14
        )

        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)
        assert isinstance(adx, pd.Series)

    def test_stochastic_rsi(self, price_data):
        """测试Stochastic RSI"""
        from factors.generators.technical import TechnicalIndicators

        stoch_k, stoch_d = TechnicalIndicators.stochastic_rsi(
            price_data['close'],
            rsi_window=14,
            stoch_window=14
        )

        assert isinstance(stoch_k, pd.Series)
        assert isinstance(stoch_d, pd.Series)


class TestDataFormatCompatibility:
    """数据格式兼容性测试"""

    def test_multiindex_format_preserved(self):
        """测试MultiIndex格式是否正确保持"""
        from factors.generators.technical import MomentumCalculator

        # 创建MultiIndex数据
        close = create_mock_price_data(n_dates=50, n_stocks=3)

        # 计算动量
        momentum = MomentumCalculator.calculate_momentum_1m(close)

        # 验证索引结构
        assert isinstance(momentum.index, pd.MultiIndex)
        assert momentum.index.names == ['TradingDates', 'StockCodes']

    def test_groupby_calculation_correctness(self):
        """测试分组计算的正确性"""
        from factors.generators.technical import MomentumCalculator

        # 创建简单的测试数据
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        stocks = ['A', 'B']
        index = pd.MultiIndex.from_product(
            [dates, stocks],
            names=['TradingDates', 'StockCodes']
        )

        # Stock A: 价格持续上涨
        # Stock B: 价格持续下跌
        prices_a = np.linspace(100, 150, 30)
        prices_b = np.linspace(100, 50, 30)
        prices = []
        for i in range(30):
            prices.extend([prices_a[i], prices_b[i]])

        close = pd.Series(prices, index=index)

        # 计算动量
        momentum = MomentumCalculator.calculate_momentum(close, period=10)

        # 获取最后一天的动量
        last_date = dates[-1]
        mom_a = momentum.loc[(last_date, 'A')]
        mom_b = momentum.loc[(last_date, 'B')]

        # A的动量应为正，B的动量应为负
        assert mom_a > 0, f"Stock A momentum should be positive, got {mom_a}"
        assert mom_b < 0, f"Stock B momentum should be negative, got {mom_b}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
