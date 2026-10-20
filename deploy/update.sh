#!/usr/bin/env bash
# Обновление UX-рецензента на сервере (/opt/ux-reviewer): свежий образ из ghcr.io →
# если он новый (или сервис не запущен) — docker compose up -d. Вручную или по cron.
set -uo pipefail
D=/opt/ux-reviewer
IMAGE=ghcr.io/johnsmith-sg/ux-reviewer:latest
LOG=$D/update.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
cd "$D" || exit 1

old=$(docker inspect -f '{{.Image}}' ux-reviewer 2>/dev/null || echo none)
if ! docker compose pull -q >/dev/null 2>&1; then
  say "ошибка: не удалось скачать $IMAGE"; exit 1
fi
new=$(docker image inspect -f '{{.Id}}' "$IMAGE")
running=$(docker inspect -f '{{.State.Running}}' ux-reviewer 2>/dev/null || echo false)
if [ "$new" = "$old" ] && [ "$running" = true ]; then
  say "без изменений: ${new:7:12}"; exit 0
fi
docker compose up -d --no-build >/dev/null 2>&1
sleep 5
say "обновлено: ${old:7:12} → ${new:7:12}, $(docker inspect -f '{{.State.Status}}' ux-reviewer)"
docker image prune -f >/dev/null 2>&1
