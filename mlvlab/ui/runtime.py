from __future__ import annotations

from typing import Any, Optional, Callable
import time
from threading import Lock, Thread
import atexit
import random  # <-- AÑADIDO: Necesario para generar la nueva semilla

from .state import StateStore
from mlvlab.core.trainer import Trainer


class SimulationRunner:
    """
    Ejecuta el bucle de simulación en un hilo dedicado, usando una clase
    de lógica interactiva para mantener la UI responsiva y ejecutar el código del alumno.
    """

    def __init__(self, trainer: Trainer, state: StateStore, env_lock: Lock, **kwargs) -> None:
        self.trainer = trainer
        self.env = self.trainer.env
        self.agent = self.trainer.agent
        self.logic = self.trainer.logic
        self.state = state
        self.env_lock = env_lock

        self._thread: Optional[Thread] = None
        self._stop = False
        self._last_check_time = time.time()
        self._steps_at_last_check = 0
        self._atexit_registered = False

        self._episode_active = False
        self._current_state = None

        with self.env_lock:
            obs, info = self.env.reset()

        self.state.set(['sim', 'obs'], obs)
        self.state.set(['sim', 'info'], info)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
        if not self._atexit_registered:
            try:
                atexit.register(self.stop)
                self._atexit_registered = True
            except Exception:
                pass

    def stop(self) -> None:
        self._stop = True
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass
        self._thread = None

    def _loop(self) -> None:
        while not self._stop:
            cmd = self.state.get(['sim', 'command']) or "run"
            if cmd == "pause":
                time.sleep(0.01)
                continue

            # --- SECCIÓN MODIFICADA ---
            if cmd == "reset":
                # 1. Generamos una nueva semilla aleatoria para cambiar el escenario
                new_seed = random.randint(0, 1_000_000)

                # 2. Reiniciamos el entorno CON la nueva semilla
                with self.env_lock:
                    obs, info = self.env.reset(seed=new_seed)

                # 3. Reiniciamos el agente y las métricas
                if hasattr(self.agent, "reset"):
                    self.agent.reset()
                self.state.set(['sim', 'current_episode_reward'], 0.0)
                self.state.set(['sim', 'total_steps'], 0)
                self.state.set(['metrics', 'episodes_completed'], 0)
                self.state.set(['metrics', 'reward_history'], [])
                self.state.set(['agent', 'epsilon'], 1.0)
                self.state.set(['sim', 'seed'], new_seed)

                # 4. Preparamos el estado inicial para el nuevo episodio
                self._current_state = self.logic._obs_to_state(obs)
                self._episode_active = True  # El episodio ya ha comenzado con este reset

                # 5. Volvemos al modo "run" y continuamos el bucle
                self.state.set(['sim', 'command'], "run")
                continue
            # --- FIN DE LA SECCIÓN MODIFICADA ---

            if not self._episode_active:
                with self.env_lock:
                    self._current_state = self.logic.on_episode_start()
                self._episode_active = True
                self.state.set(['sim', 'current_episode_reward'], 0.0)

            try:
                if hasattr(self.agent, 'learning_rate'):
                    lr_state = self.state.get(['agent', 'learning_rate'])
                    if lr_state is not None:
                        self.agent.learning_rate = float(lr_state)
                if hasattr(self.agent, 'discount_factor'):
                    df_state = self.state.get(['agent', 'discount_factor'])
                    if df_state is not None:
                        self.agent.discount_factor = float(df_state)
                if hasattr(self.agent, 'epsilon'):
                    eps_state = self.state.get(['agent', 'epsilon'])
                    if eps_state is not None:
                        self.agent.epsilon = float(eps_state)
            except Exception:
                pass

            with self.env_lock:
                next_state, reward, done = self.logic.step(self._current_state)

            self._current_state = next_state

            cur_rew = float(self.state.get(
                ['sim', 'current_episode_reward']) or 0.0) + reward
            self.state.set(['sim', 'current_episode_reward'],
                           round(cur_rew, 2))
            total_steps = int(self.state.get(['sim', 'total_steps']) or 0) + 1
            self.state.set(['sim', 'total_steps'], total_steps)

            if done:
                self._episode_active = False
                with self.env_lock:
                    self.logic.on_episode_end()

                episodes = int(self.state.get(
                    ['metrics', 'episodes_completed']) or 0) + 1
                self.state.set(['metrics', 'episodes_completed'], episodes)
                history = list(self.state.get(
                    ['metrics', 'reward_history']) or [])
                history.append([episodes, round(cur_rew, 2)])
                max_len = int(self.state.get(
                    ['metrics', 'chart_reward_number']) or 100)
                if len(history) > max_len:
                    history = history[-max_len:]
                self.state.set(['metrics', 'reward_history'], history)

                if hasattr(self.agent, 'epsilon'):
                    self.state.set(['agent', 'epsilon'], self.agent.epsilon)

            spm = max(1, int(self.state.get(['sim', 'speed_multiplier']) or 1))
            turbo = bool(self.state.get(['sim', 'turbo_mode']) or False)

            if not turbo:
                sleep_duration = 1.0 / spm
                time.sleep(sleep_duration)

            now = time.time()
            elapsed = now - self._last_check_time
            if elapsed > 0.5:
                steps_this_interval = total_steps - self._steps_at_last_check
                sps = int(steps_this_interval / elapsed) if elapsed > 0 else 0
                self.state.set(['metrics', 'steps_per_second'], sps)
                self._last_check_time = now
                self._steps_at_last_check = total_steps
