# mlvlab/ui/runtime.py

from __future__ import annotations

from typing import Optional
import time
from threading import Lock, Thread
import atexit
import random
import os  # <<< CAMBIO: Importamos 'os' para contar los núcleos de la CPU

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
        self._runner_state = "RUNNING"

        # Detectamos si la CPU es de bajo rendimiento al iniciar.
        # Consideramos "bajo rendimiento" 4 núcleos o menos.
        try:
            self._is_low_power = (os.cpu_count() or 4) <= 4
        except NotImplementedError:
            # Si no se puede detectar, asumimos que sí.
            self._is_low_power = True

        # Simplemente inicializamos los valores del estado a None o por defecto.
        # Podemos generar una semilla para mostrar
        initial_seed = random.randint(0, 1_000_000)
        self.state.set(['sim', 'seed'], initial_seed)
        self.state.set(['sim', 'obs'], None)
        self.state.set(['sim', 'info'], {})

    def start(self) -> None:
        # Esta función no cambia
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
        # Esta función no cambia
        self._stop = True
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass
        self._thread = None

    def _loop(self) -> None:
        # <<< CAMBIO: Añadimos un contador de pasos para ceder el control en modo turbo
        steps_since_yield = 0

        while not self._stop:
            if self._runner_state == "ENDING_SCENE":
                is_finished = True
                if hasattr(self.env.unwrapped, "is_end_scene_animation_finished"):
                    with self.env_lock:
                        is_finished = self.env.unwrapped.is_end_scene_animation_finished()
                if is_finished:
                    self._episode_active = False
                    self._runner_state = "RUNNING"
                    cur_rew = float(self.state.get(
                        ['sim', 'current_episode_reward']) or 0.0)
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
                        self.state.set(['agent', 'epsilon'],
                                       self.agent.epsilon)
                else:
                    time.sleep(1/60)
                    continue

            cmd = self.state.get(['sim', 'command']) or "run"
            if cmd == "pause":
                time.sleep(0.01)
                continue

            # ... (código del 'reset' sin cambios)
            if cmd == "reset":
                self._runner_state = "RUNNING"
                new_seed = random.randint(0, 1_000_000)
                with self.env_lock:
                    obs, info = self.env.reset(seed=new_seed)
                if hasattr(self.agent, "reset"):
                    self.agent.reset()

                self.state.set(['sim', 'current_episode_reward'], 0.0)
                self.state.set(['sim', 'total_steps'], 0)
                self.state.set(['metrics', 'episodes_completed'], 0)
                self.state.set(['metrics', 'reward_history'], [])
                self.state.set(['agent', 'epsilon'], 1.0)
                self.state.set(['sim', 'seed'], new_seed)

                # Incrementamos el contador de resets (Nueva generación).
                current_reset_count = int(self.state.get(
                    ['sim', 'reset_counter']) or 0) + 1
                self.state.set(['sim', 'reset_counter'], current_reset_count)

                self._current_state = self.logic._obs_to_state(obs)
                self._episode_active = True
                self.state.set(['sim', 'command'], "run")

                # Lógica de espera sincronizada robusta
                turbo = bool(self.state.get(['sim', 'turbo_mode']) or False)
                if not turbo:
                    target_reset_count = current_reset_count
                    start_wait = time.time()
                    while True:
                        # Esperamos hasta que el renderer confirme que está en la generación actual.
                        rendered_reset_count = int(self.state.get(
                            ['ui', 'last_rendered_reset_counter']) or -1)
                        if rendered_reset_count >= target_reset_count:
                            break
                        if time.time() - start_wait > 0.5:  # Timeout 0.5s
                            break
                        time.sleep(0.001)  # Ceder ejecución
                    continue  # Forzar re-evaluación de la UI
                continue

            if not self._episode_active:
                with self.env_lock:
                    self._current_state = self.logic.on_episode_start()
                self._episode_active = True
                self.state.set(['sim', 'current_episode_reward'], 0.0)

                if not self.state.get(['sim', 'initialized']):
                    self.state.set(['sim', 'initialized'], True)

                current_reset_count = int(self.state.get(
                    ['sim', 'reset_counter']) or 0) + 1
                self.state.set(['sim', 'reset_counter'], current_reset_count)

                turbo = bool(self.state.get(['sim', 'turbo_mode']) or False)
                if not turbo:
                    target_reset_count = current_reset_count
                    start_wait = time.time()
                    while True:
                        rendered_reset_count = int(self.state.get(
                            ['ui', 'last_rendered_reset_counter']) or -1)
                        if rendered_reset_count >= target_reset_count:
                            break
                        if time.time() - start_wait > 0.5:  # Timeout 0.5s
                            break
                        time.sleep(0.001)
                    continue

            spm = max(1, int(self.state.get(['sim', 'speed_multiplier']) or 1))
            turbo = bool(self.state.get(['sim', 'turbo_mode']) or False)

            # Lógica para ajustar la velocidad del modo turbo
            if turbo:
                effective_speed = 10000.0 if self._is_low_power else 100000.0
            else:
                effective_speed = float(spm)

            if hasattr(self.env.unwrapped, "set_simulation_speed"):
                self.env.unwrapped.set_simulation_speed(effective_speed)

            try:
                if hasattr(self.agent, 'learning_rate'):
                    lr_state = self.state.get(["agent", "learning_rate"]) or getattr(
                        self.agent, 'learning_rate')
                    setattr(self.agent, 'learning_rate', float(lr_state))
                if hasattr(self.agent, 'discount_factor'):
                    df_state = self.state.get(["agent", "discount_factor"]) or getattr(
                        self.agent, 'discount_factor')
                    setattr(self.agent, 'discount_factor', float(df_state))
                if hasattr(self.agent, 'epsilon'):
                    eps_state = self.state.get(
                        ["agent", "epsilon"]) or getattr(self.agent, 'epsilon')
                    setattr(self.agent, 'epsilon', float(eps_state))
            except Exception:
                pass

            with self.env_lock:
                next_state, reward, done, info = self.logic.step(
                    self._current_state)
                if info and spm <= 200 and not turbo and 'play_sound' in info and info['play_sound']:
                    self.state.set(['sim', 'last_sound'], info['play_sound'])

            self._current_state = next_state
            cur_rew = float(self.state.get(
                ['sim', 'current_episode_reward']) or 0.0) + reward
            self.state.set(['sim', 'current_episode_reward'],
                           round(cur_rew, 2))
            total_steps = int(self.state.get(['sim', 'total_steps']) or 0) + 1
            self.state.set(['sim', 'total_steps'], total_steps)

            total_steps = int(self.state.get(['sim', 'total_steps']) or 0) + 1
            self.state.set(['sim', 'total_steps'], total_steps)
            if done:
                turbo = bool(self.state.get(['sim', 'turbo_mode']) or False)

                if self.trainer.use_end_scene_animation and effective_speed <= 50:
                    self.logic.on_episode_end()
                    self._runner_state = "ENDING_SCENE"
                else:
                    if not turbo:
                        target_step = total_steps
                        start_wait = time.time()
                        while True:
                            rendered_step = int(self.state.get(
                                ['ui', 'last_frame_step']) or 0)
                            if rendered_step >= target_step:
                                break
                            if time.time() - start_wait > 0.5:
                                break
                            time.sleep(0.001)
                    self._episode_active = False
                    cur_rew_done = float(self.state.get(
                        ['sim', 'current_episode_reward']) or 0.0)
                    episodes_done = int(self.state.get(
                        ['metrics', 'episodes_completed']) or 0) + 1
                    self.state.set(
                        ['metrics', 'episodes_completed'], episodes_done)
                    history_done = list(self.state.get(
                        ['metrics', 'reward_history']) or [])
                    history_done.append(
                        [episodes_done, round(cur_rew_done, 2)])
                    max_len_done = int(self.state.get(
                        ['metrics', 'chart_reward_number']) or 100)
                    if len(history_done) > max_len_done:
                        history_done = history_done[-max_len_done:]
                    self.state.set(['metrics', 'reward_history'], history_done)
                    if hasattr(self.agent, 'epsilon'):
                        self.state.set(['agent', 'epsilon'],
                                       self.agent.epsilon)

            # Lógica para ceder control y mantener la UI responsiva
            if turbo:
                steps_since_yield += 1
                # Cada 1000 pasos, hacemos una pausa minúscula para que la UI respire.
                # Este número (1000) se puede ajustar.
                if steps_since_yield >= 1000:
                    time.sleep(0)
                    steps_since_yield = 0
            else:
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
