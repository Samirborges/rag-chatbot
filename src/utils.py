import os


def get_env_var(name: str) -> str:
    """Lê uma variável de ambiente obrigatória, falhando cedo se não existir."""
    try:
        return os.environ[name]
    except KeyError:
        raise EnvironmentError(f"Variável de ambiente obrigatória não definida: {name}") from None