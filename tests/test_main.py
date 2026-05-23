from main import main


def test_main_prints_project_greeting(capsys):
    main()

    assert capsys.readouterr().out == "Hello from ai-nids!\n"
