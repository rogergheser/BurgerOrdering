import pytest
from components import (
    PreNLU,
    NLU,
    DM,
    NLG
)

class Tester():
    def __init__(self, 
                 pre_nlu: PreNLU,
                 nlu: NLU,
                 dm: DM,
                 nlg: NLG,
                 cfg: dict):
        """
        :param pre_nlu: PreNLU component to be tested
        :param nlu: NLU component to be tested
        :param dm: DM component to be tested
        :param nlg: NLG component to be tested
        :param cfg: Configuration file for the Tester
        """
        self.pre_nlu = pre_nlu
        self.nlu = nlu
        self.dm = dm
        self.nlg = nlg
        self.cfg = cfg


    def test_pre_nlu(self):
        """
        Test the PreNLU component.
        """
        pass

    def test_nlu(self):
        """
        Test the NLU component.
        """
        