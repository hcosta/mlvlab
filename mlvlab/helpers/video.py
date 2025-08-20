from __future__ import annotations

import os
from pathlib import Path
from typing import List
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx.MultiplySpeed import MultiplySpeed
from mlvlab.i18n.core import i18n


def merge_videos_with_counter(
    video_folder: str | os.PathLike,
    output_filename: str | os.PathLike,
    font_path: str | os.PathLike | None = None,
    cleanup: bool = True,
    speed_multiplier: float = 1.0,
) -> bool:
    """
    Une todos los vídeos MP4 en `video_folder` en orden alfanumérico y añade un contador
    (i/N) como overlay en la esquina inferior-derecha. Devuelve True si se generó
    correctamente el archivo final.

    Compatibilidad con MoviePy 2.2.1, usando `.with_position` y `.with_duration`.
    """
    folder = str(video_folder)
    output_path = str(output_filename)

    video_files: List[str] = [
        f for f in os.listdir(folder) if f.endswith('.mp4')]
    video_files.sort()
    if not video_files:
        print(i18n.t("helpers.video.no_videos_found"))
        return False

    clips_originales = []
    clips_con_texto = []
    try:
        total_episodes = len(video_files)
        print(i18n.t("helpers.video.processing_videos", total_episodes=total_episodes))
        for i, filename in enumerate(video_files, 1):
            filepath = os.path.join(folder, filename)
            try:
                clip = VideoFileClip(filepath)
                if speed_multiplier != 1.0:
                    # 1. Creamos una instancia del efecto con el factor de velocidad
                    speed_effect = MultiplySpeed(factor=speed_multiplier)
                    # 2. Aplicamos el efecto al clip usando su método .apply()
                    clip = speed_effect.apply(clip)
                clips_originales.append(clip)

                texto = f"{i}/{total_episodes}  \n"
                if font_path is not None:
                    txt_clip = TextClip(text=texto, font=str(
                        font_path), font_size=32, color='white')
                else:
                    txt_clip = TextClip(
                        text=texto, font_size=32, color='white')

                txt_clip = txt_clip.with_position(
                    ('right', 'top')).with_duration(clip.duration)
                video_con_texto = CompositeVideoClip([clip, txt_clip])
                clips_con_texto.append(video_con_texto)
            except Exception as e:
                print(i18n.t("helpers.video.error_processing_clip", filename=filename, error=str(e)))

        if not clips_con_texto:
            print(i18n.t("helpers.video.error_processing_videos"))
            return False

        print(i18n.t("helpers.video.merging_videos", count=len(clips_con_texto), output_path=output_path))
        final_clip = concatenate_videoclips(clips_con_texto, method='compose')
        final_clip.write_videofile(output_path, logger='bar')
    except Exception as e:
        print(i18n.t("helpers.video.error_creating_video", error=str(e)))
    finally:
        for clip in clips_con_texto:
            try:
                clip.close()
            except Exception:
                pass
        for clip in clips_originales:
            try:
                clip.close()
            except Exception:
                pass

    ok = os.path.exists(output_path)
    if ok:
        print(i18n.t("helpers.video.video_created_success"))
        if cleanup:
            for f in video_files:
                try:
                    os.remove(os.path.join(folder, f))
                except Exception:
                    pass
    else:
        print(i18n.t("helpers.video.error_final_video"))
    return ok
