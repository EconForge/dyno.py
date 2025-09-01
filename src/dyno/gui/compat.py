def fix_solara():
    ###
    ### workaround to import solara
    ###

    from mock import MagicMock
    import sys

    sys.modules["fcntl"] = MagicMock()
    sys.modules["nest_asyncio"] = MagicMock()
    sys.modules["requests"] = MagicMock()

    import ipyvuetify

    sys.modules["ipyvuetify.extra"] = MagicMock()

    import comm

    old = comm.create_comm
    comm.create_comm = None
    comm._create_comm = "Not None"

    import solara

    comm.create_comm = old
    comm._create_comm = None

    try:
        del sys.modules["solara.server"]
    except:
        pass

    def _using_solara_server():
        return False

    solara.toestand._using_solara_server = _using_solara_server
