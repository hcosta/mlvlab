import gymnasium as gym
import mlvlab  # ¡Importante! Esto registra los entornos "mlv/..." y mantiene compatibilidad con los antiguos

# Crea el entorno como lo harías normalmente con Gymnasium
env = gym.make("mlv/ant-v1", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    # Aquí es donde va tu lógica para elegir una acción
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()


""""""""" DEBERIA LANZAR UNA VENTANAAAAAAAAAA """"""""
