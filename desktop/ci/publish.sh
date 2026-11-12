#!/usr/bin/env bash
set -euo pipefail

cd dist
windows=MOSS-TTS-Nano-Windows-x86_64.zip
linux=MOSS-TTS-Nano-Linux-x86_64.tar.gz
sha256sum "$windows" "$linux" > SHA256SUMS.txt
assets=("$windows" "$linux" SHA256SUMS.txt)

if [[ "$GITHUB_REF_TYPE" == tag ]]; then
  tag=$GITHUB_REF_NAME
  revision=$(gh api "repos/$GH_REPO/commits/$GITHUB_SHA" --jq '.sha')
  tagged_revision=$(gh api "repos/$GH_REPO/commits/$tag" --jq '.sha')
  if [[ "$tagged_revision" != "$revision" ]]; then
    echo "Tag $tag moved after this build." >&2
    exit 1
  fi
  target=(--verify-tag)
else
  tag="desktop-$GITHUB_RUN_ID"
  target=(--target "$GITHUB_SHA")
fi

if draft=$(gh release view "$tag" --json isDraft --jq '.isDraft' 2>/dev/null); then
  if [[ "$draft" == true ]]; then
    gh release upload "$tag" "${assets[@]}" --clobber
    gh release edit "$tag" --draft=false --latest
  else
    names=$(gh release view "$tag" --json assets --jq '.assets[].name')
    for asset in "${assets[@]}"; do
      if ! grep -Fxq -- "$asset" <<< "$names"; then
        echo "Published release $tag is missing $asset." >&2
        exit 1
      fi
    done
  fi
else
  gh release create "$tag" "${assets[@]}" "${target[@]}" --latest --generate-notes \
    --title "MOSS-TTS Nano · $tag" \
    --notes "Download the Windows ZIP or Linux tar.gz below, extract the entire archive, and open MOSS-TTS. Both packages include the ONNX models for offline use."
fi

printf '### 便携版下载\n\n[打开 GitHub Release](%s/%s/releases/tag/%s)\n' \
  "$GITHUB_SERVER_URL" "$GH_REPO" "$tag" >> "$GITHUB_STEP_SUMMARY"
