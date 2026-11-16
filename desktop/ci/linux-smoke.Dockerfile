ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG DISPLAY_BACKEND
RUN if command -v apt-get >/dev/null; then \
      apt-get update && apt-get install -y --no-install-recommends \
        bash tar gzip fonts-dejavu-core libegl1 libgl1 libfontconfig1 libpulse0 libdbus-1-3 \
        $(if [ "$DISPLAY_BACKEND" = wayland ]; then echo weston; else echo xvfb; fi); \
    elif command -v dnf >/dev/null; then \
      dnf install -y bash tar gzip dejavu-sans-fonts libglvnd-egl libglvnd-glx fontconfig \
        pulseaudio-libs dbus-libs xorg-x11-server-Xvfb; \
    else \
      pacman -Syu --noconfirm bash tar gzip ttf-dejavu libglvnd fontconfig libpulse dbus xorg-server-xvfb; \
    fi
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix
COPY linux-smoke.sh /linux-smoke.sh
ENTRYPOINT ["bash", "/linux-smoke.sh"]
