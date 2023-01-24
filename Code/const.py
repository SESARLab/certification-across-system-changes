import enum
import math
import typing

CHANGE_CAUSE_NO = 0
CHANGE_CAUSE_ENV = 1
CHANGE_CAUSE_CODE = 2
CHANGE_CAUSE_CODE_ENV = 3
# not used, but in situation we add all.
CHANGE_CAUSE_VULN = 4

CRITICAL_FALSE = 0
CRITICAL_TRUE = 1

NOVELTY_FALSE = 0
NOVELTY_TRUE = 1

CODE_EXTENT_NO = -1
CODE_EXTENT_MINOR = 0
CODE_EXTENT_MAJOR = 1

PREFIX_GT = 'GT'
PREFIX_OUR = 'OUR'
PREFIX_STOTA = 'STOTA'

PREFIX_MODEL = 'Model'

COMP_PREFIX = 'COMP'

COLUMN_NAME_CRITICAL = f'{PREFIX_GT}(Crit)'
COLUMN_NAME_CHANGE_TYPE = f'{PREFIX_GT}(Type)'
COLUMN_NAME_CODE_EXTENT = f'{PREFIX_GT}(Code)'
COLUMN_NAME_CHANGED_COMP = f'{PREFIX_GT}(Change)'
COLUMN_NAME_CHANGED_COMP_PRIMARY = f'{PREFIX_GT}(PrimaryChange)'
COLUMN_NAME_CHANGED_COMP_SECONDARY = f'{PREFIX_GT}(SecondaryChange)'
# always set to False.
COLUMN_NAME_NOVELTY = f'{PREFIX_GT}(Novel)'

# contains the columns not containing components name.
COLUMN_NAMES_NO_COMP = [COLUMN_NAME_CHANGE_TYPE, COLUMN_NAME_CRITICAL, COLUMN_NAME_CODE_EXTENT,
                        COLUMN_NAME_CHANGED_COMP, COLUMN_NAME_CHANGED_COMP_PRIMARY,
                        COLUMN_NAME_CHANGED_COMP_SECONDARY, COLUMN_NAME_NOVELTY]

# PREFIXES = [PREFIX_OUR, PREFIX_STOTA]

PREFIX_OUR_STOTA = [PREFIX_OUR, PREFIX_STOTA]
PREFIX_ALL = [PREFIX_GT, PREFIX_OUR, PREFIX_STOTA]

COLUMN_NAME_CHANGE_TYPE_OUR = f'{PREFIX_OUR}({COLUMN_NAME_CHANGE_TYPE})'
COLUMN_NAME_CODE_EXTENT_OUR = f'{PREFIX_OUR}({COLUMN_NAME_CODE_EXTENT})'
COLUMN_NAME_CRITICAL_OUR = f'{PREFIX_OUR}({COLUMN_NAME_CRITICAL})'
COLUMN_NAME_CHANGED_COMP_OUR = f'{PREFIX_OUR}({COLUMN_NAME_CHANGED_COMP})'
COLUMN_NAME_NOVELTY_OUR = f'{PREFIX_OUR}({COLUMN_NAME_NOVELTY})'

COLUMN_NAME_CHANGE_TYPE_STOTA = f'{PREFIX_STOTA}({COLUMN_NAME_CHANGE_TYPE})'
COLUMN_NAME_CODE_EXTENT_STOTA = f'{PREFIX_STOTA}({COLUMN_NAME_CODE_EXTENT})'
COLUMN_NAME_CRITICAL_STOTA = f'{PREFIX_STOTA}({COLUMN_NAME_CRITICAL})'
COLUMN_NAME_CHANGED_COMP_STOTA = f'{PREFIX_STOTA}({COLUMN_NAME_CHANGED_COMP})'
COLUMN_NAME_NOVELTY_STOTA = f'{PREFIX_STOTA}({COLUMN_NAME_NOVELTY})'

COLUMN_NAME_SITUATION = 'S'


def prefix(pre: str, col_name: str):
    return f'{pre}({col_name})'


class EvalType(enum.Enum):
    GT = PREFIX_GT
    OUR = PREFIX_OUR
    STOTA = PREFIX_STOTA

    def col_name(self, column_name: str):
        if self == EvalType.OUR or self == EvalType.STOTA:
            return f'{self.name}({column_name})'
        else:
            if column_name.startswith(self.name):
                return column_name
            else:
                return f'{self.name}({column_name})'

    def remove_prefixes(self, columns: typing.List[typing.Union[str, int]]) -> typing.List[typing.Union[str, int]]:
        new_cols = []
        for col_name in columns:
            if isinstance(col_name, int):
                new_cols.append(col_name)
            else:
                col_name: str = col_name
                if self == EvalType.OUR or self == EvalType.STOTA:
                    if col_name.startswith(self.name):
                        new_col_name = col_name.replace(f'{self.name}(', '').removesuffix(')')
                        new_cols.append(new_col_name)
                    else:
                        new_cols.append(col_name)
                else:
                    new_cols.append(col_name)
        return new_cols

    def get_pertinent_columns(self, columns: typing.List[typing.Union[str, int]],
                              include_components: bool = False):
        new_cols = []
        for col in columns:
            if isinstance(col, int):
                if include_components:
                    new_cols.append(col)
            else:
                col: str = col
                if col.startswith(f'{self.name}('):
                    new_cols.append(col)
        return new_cols


class DataType(enum.Enum):
    RAW = 'RAW'
    NORMALIZED = 'NORMALIZED'


def PREFIX_FUNC_OUR(col: str) -> str:
    return f'{PREFIX_OUR}({col})'


def PREFIX_FUNC_STOTA(col: str) -> str:
    return f'{PREFIX_STOTA}({col})'


def get_formatter(size: int):
    """
    Given a maximum size e.g., 100 we return
    an adequate formatter such that any number less than size is formatted the same way.

    Examples
    -------
    >>> formatter = get_formatter(100)
    >>> formatter.format(99)
    099
    """
    n_unit = int(math.log10(size))
    return '{:0' + str(n_unit) + '}'


OUTPUT_COLUMN_NAME_CORRECT_COMPONENTS = 'CorrectComp'

OUTPUT_COLUMN_NAME_ACC = 'Acc'
OUTPUT_COLUMN_NAME_PREC = 'Pre'
OUTPUT_COLUMN_NAME_REC = 'Rec'
OUTPUT_COLUMN_NAME_F1 = 'F1'

OUTPUT_COLUMN_NAME_RECERT_NO = 'ReCert_No'
OUTPUT_COLUMN_NAME_RECERT_PARTIAL = 'No_ReCert_Partial'
OUTPUT_COLUMN_NAME_RECERT_FULL = 'ReCert_Full'

OUTPUT_COLUMN_NAME_CHANGE_TYPE_ALL = 'ChangeType_All'
OUTPUT_COLUMN_NAME_CHANGE_TYPE_ENV = 'ChangeType_Env'
OUTPUT_COLUMN_NAME_CHANGE_TYPE_CODE = 'ChangeType_Code'
OUTPUT_COLUMN_NAME_CHANGE_TYPE_CODE_ENV = 'ChangeType_CodeEnv'

OUTPUT_COLUMN_NAME_S0_OK = 'S0_Ok'
OUTPUT_COLUMN_NAME_S1_OK = 'S1_Ok'
OUTPUT_COLUMN_NAME_S2_OK = 'S2_Ok'
OUTPUT_COLUMN_NAME_S3_OK = 'S3_Ok'
# OUTPUT_COLUMN_NAME_SLESS_OK = 'S*_Ok'

OUTPUT_COLUMN_COMPONENT_PER_SITUATION_COUNT = 'CorrectComponents'


def apply_col_name(base: str, column_name: str) -> str:
    return f'{base}({column_name})'
