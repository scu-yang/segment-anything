from enum import Enum
from enum import unique


@unique
class ErrorCode(Enum):
    NOOP = ("000000", "success")
    FAILURE = ("100000", "failure")
    PARAM_ERROR = ("400001", "parameter error.")
    PARAM_NOT_REQUIRED = ("400001", "parameter {} not require.")

    def __init__(self, code, msg):
        self.code = code
        self.msg = msg


class ServerException(Exception):
    def __init__(self, errorCode: ErrorCode, *params):
        if params is not None:
            msg = str.format(errorCode.msg, params)
        else:
            msg = errorCode.msg
        super().__init__(errorCode.code, msg)
        self.message = msg
        self.status = errorCode.code