"""
风险模型异常定义

定义风险模型模块中使用的所有异常类
"""


class RiskModelError(Exception):
    """
    风险模型异常基类
    
    所有风险模型相关的异常都继承自此类
    """
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ModelNotFittedError(RiskModelError):
    """
    模型未拟合异常
    
    当尝试使用未拟合的模型进行预测或计算时抛出
    """
    def __init__(self, model_name: str = None):
        message = f"Model {model_name or 'risk model'} has not been fitted yet. Call fit() first."
        super().__init__(message, "MODEL_NOT_FITTED")


class SingularCovarianceError(RiskModelError):
    """
    协方差矩阵奇异异常
    
    当协方差矩阵不可逆或数值不稳定时抛出
    """
    def __init__(self, matrix_name: str = "covariance matrix", condition_number: float = None):
        if condition_number:
            message = f"{matrix_name} is singular or ill-conditioned (condition number: {condition_number:.2e})"
        else:
            message = f"{matrix_name} is singular or ill-conditioned"
        super().__init__(message, "SINGULAR_COVARIANCE")


class OptimizationConvergenceError(RiskModelError):
    """
    优化收敛失败异常
    
    当组合优化算法无法收敛时抛出
    """
    def __init__(self, optimizer_name: str = None, iterations: int = None, message: str = None):
        if message:
            error_message = message
        elif optimizer_name and iterations:
            error_message = f"{optimizer_name} failed to converge after {iterations} iterations"
        else:
            error_message = "Optimization failed to converge"
        super().__init__(error_message, "OPTIMIZATION_CONVERGENCE")


class InsufficientDataError(RiskModelError):
    """
    数据不足异常
    
    当数据量不足以进行可靠的风险建模时抛出
    """
    def __init__(self, required_observations: int = None, available_observations: int = None, 
                 data_type: str = "data"):
        if required_observations and available_observations:
            message = (f"Insufficient {data_type}: need at least {required_observations} observations, "
                      f"but only {available_observations} available")
        else:
            message = f"Insufficient {data_type} for reliable risk modeling"
        super().__init__(message, "INSUFFICIENT_DATA")


class InvalidParameterError(RiskModelError):
    """
    无效参数异常
    
    当传入的参数值无效或超出范围时抛出
    """
    def __init__(self, parameter_name: str, value, valid_range: str = None):
        if valid_range:
            message = f"Invalid value for parameter '{parameter_name}': {value}. Valid range: {valid_range}"
        else:
            message = f"Invalid value for parameter '{parameter_name}': {value}"
        super().__init__(message, "INVALID_PARAMETER")


class DataFormatError(RiskModelError):
    """
    数据格式异常
    
    当输入数据格式不符合要求时抛出
    """
    def __init__(self, expected_format: str, actual_format: str = None):
        if actual_format:
            message = f"Data format error: expected {expected_format}, got {actual_format}"
        else:
            message = f"Data format error: expected {expected_format}"
        super().__init__(message, "DATA_FORMAT")


class CalculationError(RiskModelError):
    """
    计算异常
    
    当风险计算过程中出现数值错误时抛出
    """
    def __init__(self, calculation_type: str, details: str = None):
        if details:
            message = f"Error in {calculation_type}: {details}"
        else:
            message = f"Error in {calculation_type}"
        super().__init__(message, "CALCULATION_ERROR")


class ConfigurationError(RiskModelError):
    """
    配置异常
    
    当模型配置不正确或冲突时抛出
    """
    def __init__(self, config_item: str, issue: str):
        message = f"Configuration error in '{config_item}': {issue}"
        super().__init__(message, "CONFIGURATION_ERROR")