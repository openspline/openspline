class OpensplineError(Exception):
    def __init__(self, message, code="error"):
        super().__init__(message)
        self.code = code


class CapacityError(OpensplineError):
    pass


class ConfigurationError(OpensplineError):
    pass


class ConnectionError(OpensplineError):
    pass


class InferenceError(OpensplineError):
    pass


def error_from(code, message):
    cls = {
        "capacity": CapacityError,
        "configuration": ConfigurationError,
        "connection": ConnectionError,
        "inference": InferenceError,
    }.get(code, OpensplineError)
    return cls(message, code)
