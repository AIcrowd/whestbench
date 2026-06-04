import scripts.sync_starterkit_docs as sk


def test_to_mdx_adds_frontmatter_and_banner_and_rewrites_links():
    md = "# Write an estimator\n\nSee [scoring](../concepts/scoring-model.md) and `<MLP>`.\n"
    out = sk.to_mdx(md, rel_slug="how-to/write-an-estimator", sha="abc1234")
    assert out.startswith('---\ntitle: "Write an estimator"')
    assert "Sourced from [whest-starterkit]" in out
    assert "abc1234" in out
    assert "/docs/participant-guide/concepts/scoring-model" in out
    assert "`<MLP>`" in out  # angle bracket inside inline code left untouched


def test_section_dirs_excludes_reference():
    assert "reference" not in sk.SECTION_DIRS
    assert "getting-started" in sk.SECTION_DIRS


def test_sanitize_escapes_prose_but_not_code():
    md = "Use <split> and {x} here.\n\n```\nkeep <raw> {braces}\n```\n\ninline `a<b` stays."
    out = sk._sanitize_mdx(md)
    assert "&lt;split>" in out
    assert "\\{x}" in out
    assert "keep <raw> {braces}" in out  # fenced code untouched
    assert "`a<b`" in out  # inline code untouched


def test_rewrite_links_same_section_bare_and_dot_slash():
    out = sk._rewrite_links("[a](sibling.md) [b](./other.md)", "getting-started/install")
    assert "/docs/participant-guide/getting-started/sibling" in out
    assert "/docs/participant-guide/getting-started/other" in out


def test_rewrite_links_cross_section_and_anchor_preserved():
    out = sk._rewrite_links("[s](../concepts/scoring-model.md#why)", "how-to/write-an-estimator")
    assert "/docs/participant-guide/concepts/scoring-model#why" in out


def test_rewrite_images_leaves_absolute_and_remote_untouched():
    out = sk._rewrite_images(
        "![a](/assets/x.svg) ![b](https://e.com/y.png)", "advanced/use", "deadbeef"
    )
    assert "(/assets/x.svg)" in out
    assert "(https://e.com/y.png)" in out
    assert "raw.githubusercontent.com" not in out


def test_local_image_rewritten_to_raw_url_and_remote_left_alone():
    md = (
        "![viz](assets/explorer.svg)\n\n"
        "![parent](../shared/diagram.png)\n\n"
        "![remote](https://cdn.example.com/x.svg)\n"
    )
    out = sk.to_mdx(md, rel_slug="advanced/use-explorer", sha="abc1234")
    # Local image resolves relative to docs/advanced/ at the pinned SHA, so no
    # local static-import target survives to break the Fumadocs/Next build.
    assert (
        "![viz](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/"
        "abc1234/docs/advanced/assets/explorer.svg)" in out
    )
    assert (
        "![parent](https://raw.githubusercontent.com/AIcrowd/whest-starterkit/"
        "abc1234/docs/shared/diagram.png)" in out
    )
    assert "![remote](https://cdn.example.com/x.svg)" in out  # remote untouched
    assert "/docs/participant-guide/assets/explorer.svg" not in out  # not namespaced
