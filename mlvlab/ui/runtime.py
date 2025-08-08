from __future__ import annotations

from typing import Any, Optional, Callable
import time
from threading import Lock, Thread
import random

from .state import StateStore


class SimulationRunner:
    """
    Ejecuta el bucle de simulación en un hilo dedicado.

    Requiere que el agente implemente al menos:
      - choose_action(state_or_obs, epsilon)
      - update(s_or_obs, action, reward, s2_or_obs, lr, gamma)
      - reset()
    """

    def __init__(self, env: Any, agent: Any, state: StateStore, env_lock: Lock, state_from_obs: Optional[Callable[[Any], Any]] = None) -> None:
        self.env = env
        self.agent = agent
        self.state = state
        self.env_lock = env_lock
        self._state_from_obs = state_from_obs

        self._thread: Optional[Thread] = None
        self._stop = False
        self._last_check_time = time.time()
        self._steps_at_last_check = 0

        # Inicialización del entorno
        with self.env_lock:
            obs, info = self.env.reset()
        self.state.set(["sim", "obs"], obs)
        self.state.set(["sim", "info"], info)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        t = self._thread
        if t and t.is_alive():
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        self._thread = None

    # --- interno --- #
    def _extract_state(self, obs: Any) -> Any:
        # Prioridad: callable pasado al runner > método del agente > obs sin transformar
        if callable(self._state_from_obs):
            try:
                return self._state_from_obs(obs)
            except Exception:
                pass
        extractor = getattr(self.agent, "extract_state_from_obs", None)
        if callable(extractor):
            try:
                return extractor(obs)
            except Exception:
                return obs
        return obs

    def _loop(self) -> None:
        last_step_time = time.perf_counter()
        step_accum = 0.0
        last_turbo = False

        while not self._stop:
            cmd = self.state.get(["sim", "command"]) or "run"
            if cmd == "pause":
                time.sleep(0.001)
                continue
            if cmd == "reset":
                # Generar nueva semilla para cambiar el escenario
                new_seed = random.randint(0, 1_000_000)
                with self.env_lock:
                    try:
                        obs, info = self.env.reset(seed=new_seed)
                    except TypeError:
                        # Fallback si el entorno no acepta seed
                        obs, info = self.env.reset()
                self.state.update("sim", {
                    "obs": obs,
                    "info": info,
                    "current_episode_reward": 0.0,
                    "total_steps": 0,
                    "seed": new_seed,
                })
                if hasattr(self.agent, "reset"):
                    try:
                        self.agent.reset()
                    except Exception:
                        pass
                # Reset de métricas
                self.state.update("metrics", {
                    "episodes_completed": 0,
                    "reward_history": [],
                    "steps_per_second": 0,
                })
                # Reset de hiperparámetros clave a valores por defecto
                self.state.update("agent", {
                    "epsilon": 1.0,
                    # mantener discount_factor y learning_rate actuales del state
                })
                # Continuar en run
                self.state.set(["sim", "command"], "run")
                continue

            now = time.perf_counter()
            dt = now - last_step_time
            # Robustez: convertir turbo a bool sin ambigüedades
            raw_turbo = self.state.get(["sim", "turbo_mode"])
            if isinstance(raw_turbo, dict) and 'value' in raw_turbo:
                raw_turbo = raw_turbo['value']
            turbo = bool(raw_turbo) if not isinstance(
                raw_turbo, str) else raw_turbo.strip().lower() in {"true", "on", "1", "yes"}
            if turbo != last_turbo:
                # Igual que en visualizer.py: al cambiar turbo, reiniciar acumulador y saltar este tick
                step_accum = 0.0
                last_step_time = now
                last_turbo = turbo
                continue
            last_step_time = now
            # sin enfriamiento adicional para respetar el comportamiento previo estable

            # Si turbo está activo, ignoramos speed_multiplier para el cálculo de steps
            spm = max(0, int(self.state.get(["sim", "speed_multiplier"]) or 1))
            if turbo:
                # Réplica de la estrategia original: muchos pasos por dt con top alto
                steps_to_do = max(
                    1, min(120000, int(30000 * dt) if dt > 0 else 10000))
            else:
                step_accum += spm * dt
                steps_to_do = int(step_accum)
            if steps_to_do <= 0:
                time.sleep(0.0005)
                continue
            steps_to_do = min(steps_to_do, 40000)

            for _ in range(steps_to_do):
                if not turbo:
                    step_accum -= 1.0
                with self.env_lock:
                    obs = self.state.get(["sim", "obs"])
                    s = self._extract_state(obs)
                    epsilon = float(self.state.get(
                        ["agent", "epsilon"]) or 0.0)
                    action = self.agent.choose_action(s, epsilon)
                    next_obs, reward, terminated, truncated, info = self.env.step(
                        action)
                    # Persistir
                    self.state.set(["sim", "obs"], next_obs)
                    self.state.set(["sim", "info"], info)
                    # Señal de audio si aplica (similar al visualizer original)
                    try:
                        speed = int(self.state.get(
                            ["sim", "speed_multiplier"]) or 1)
                        if isinstance(info, dict) and ("play_sound" in info) and (not turbo) and speed <= 50:
                            self.state.set(
                                ["sim", "last_sound"], info["play_sound"])
                    except Exception:
                        pass
                s2 = self._extract_state(next_obs)
                lr = float(self.state.get(["agent", "learning_rate"]) or 0.1)
                gamma = float(self.state.get(
                    ["agent", "discount_factor"]) or 0.9)
                self.agent.update(s, action, reward, s2, lr, gamma)

                # Métricas
                cur_rew = float(self.state.get(
                    ["sim", "current_episode_reward"]) or 0.0) + float(reward)
                self.state.set(
                    ["sim", "current_episode_reward"], round(cur_rew, 2))
                total_steps = int(self.state.get(
                    ["sim", "total_steps"]) or 0) + 1
                self.state.set(["sim", "total_steps"], total_steps)

                if terminated or truncated:
                    episodes = int(self.state.get(
                        ["metrics", "episodes_completed"]) or 0) + 1
                    self.state.set(["metrics", "episodes_completed"], episodes)
                    history = list(self.state.get(
                        ["metrics", "reward_history"]) or [])
                    history.append(round(cur_rew, 2))
                    # recorte por history_size si existe en metrics.chart_reward_number
                    max_len = int(self.state.get(
                        ["metrics", "chart_reward_number"]) or 100)
                    if len(history) > max_len:
                        history = history[-max_len:]
                    self.state.set(["metrics", "reward_history"], history)
                    with self.env_lock:
                        obs, info = self.env.reset()
                        self.state.set(["sim", "obs"], obs)
                        self.state.set(["sim", "info"], info)
                    self.state.set(["sim", "current_episode_reward"], 0.0)
                    eps = float(self.state.get(["agent", "epsilon"]) or 1.0)
                    min_eps = float(self.state.get(
                        ["agent", "min_epsilon"]) or 0.1)
                    decay = float(self.state.get(
                        ["agent", "epsilon_decay"]) or 0.99)
                    if eps > min_eps:
                        self.state.set(["agent", "epsilon"],
                                       max(min_eps, eps * decay))
                    break

            # Steps per second
            now2 = time.time()
            elapsed = now2 - self._last_check_time
            if elapsed > 0.5:
                steps_this_interval = int(self.state.get(
                    ["sim", "total_steps"]) or 0) - self._steps_at_last_check
                sps = int(steps_this_interval / elapsed) if elapsed > 0 else 0
                self.state.set(["metrics", "steps_per_second"], sps)
                self._last_check_time = now2
                self._steps_at_last_check = int(
                    self.state.get(["sim", "total_steps"]) or 0)
            time.sleep(0.0005)
