class ParserError(Exception):

    line: int
    column: int
    message: str
    details: str | None

from lark.exceptions import UnexpectedInput
from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF
# Lark parser error
# All errors inherit from lark.exceptions.UnexpectedInput
# has get context method to pretty print the problem
# UnexpectedCharacters: The lexer encountered an unexpected string
# UnexpectedToken: The parser received an unexpected token
# UnexpectedEOF: The parser expected a token, but the input ended

class LARKParserError(ParserError):
    
    def __init__(self, lark_error: UnexpectedInput, txt=None) -> None:

        line = lark_error.line
        column = lark_error.column
        if isinstance(lark_error, UnexpectedCharacters):
            message = f"Unexpected characters at {(line,column)}"
        elif isinstance(lark_error, UnexpectedToken):
            value = lark_error.token.value
            if '\n' in value:
                value = 'newline'
            message = f"Unexpected token `{value}` at {(line,column)}"
        elif isinstance(lark_error, UnexpectedEOF):
            message = f"Unexpected end of file at {(line,column)}"
        else:
            message = str(type(lark_error))

        # self.message = message
        self.column = column
        self.line = line
        if txt is not None:
            self.details = lark_error.get_context(txt)
            self.details += str(lark_error)
        else:
            self.details = str(lark_error)
        super().__init__(message)




from dynare_preprocessor import PreprocessorException, UnsupportedFeatureException

class DynareParserError(ParserError):
    
    def __init__(self, err: PreprocessorException) -> None:
        self.line = None
        self.column = None
        self.details = None
        super().__init__(str(err))

class SteadyStateError(Exception):
    
    def __init__(self, residuals) -> None:
        self.residuals = residuals
        message = f"Steady state values don't satisfy model equations. <ax residual is {max(abs(r) for r in residuals)}"
        self.details = f"Residuals: {residuals}"
        super().__init__(message)