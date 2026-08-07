# Publishing the project page

`docs/index.html` is a single self-contained file (no build step, no dependencies) plus the
figures in `docs/assets/`. Any static host can serve it. Three routes, cheapest first.

## 1. Project page on this repo — 2 minutes, no new accounts

**Do this once** (needs admin on the repo): Repository →
**Settings → Pages → Build and deployment → Source: `GitHub Actions`**.
Then Actions tab → the failed *Deploy project page* run → **Re-run all jobs**.

This step is not optional under the `ielab` org. The workflow asks to create the Pages site
itself (`configure-pages` with `enablement: true`), but `GITHUB_TOKEN` is not permitted to
create Pages sites here — the run fails with *"Create Pages site failed: Resource not
accessible by integration"*. Creating the site by hand is the only way past it; after that
the token's `pages: write` is enough to deploy, and every later push just works.

Once the site exists: `.github/workflows/pages.yml` publishes `docs/` on every push to
`main`. The page appears at:

```
https://ielab.github.io/skim-search-agent/
```

(If you prefer no workflow, choose *Deploy from a branch* → `main` → `/docs` instead. Same
result; the workflow exists so the deploy is visible in the Actions tab and can be re-run.)

## 2. The vanity URL — `skimsearchagent.github.io`

GitHub serves `https://<name>.github.io/` from a repository **named exactly**
`<name>.github.io`, owned by a user or organisation **named** `<name>`. So this URL requires
an account or organisation called `skimsearchagent` — it cannot be produced from the
`ielab/skim-search-agent` repo alone.

Steps:

1. Check availability of the name at `https://github.com/skimsearchagent`. If it 404s the
   name is free.
2. Create a **free organisation** called `skimsearchagent`
   (github.com/organizations/plan) — an organisation is better than a personal account here:
   the lab can co-own it and it survives people moving on.
3. In that organisation create a **public** repository named `skimsearchagent.github.io`.
4. Publish the page at that repository's root:

   ```bash
   git clone https://github.com/skimsearchagent/skimsearchagent.github.io
   cd skimsearchagent.github.io
   cp -r /path/to/skim-search-agent/docs/* .      # index.html, assets/, .nojekyll
   git add -A && git commit -m "publish project page" && git push
   ```

5. Settings → Pages → Source: *Deploy from a branch* → `main` → `/ (root)`.

Live within a minute at `https://skimsearchagent.github.io/`.

### Keeping the mirror in sync automatically

Optional. In the `skimsearchagent` organisation create a fine-grained personal access token
with *Contents: read and write* on `skimsearchagent.github.io`, add it to **this** repository
as the secret `PAGES_MIRROR_TOKEN`, then add this job to `.github/workflows/pages.yml`:

```yaml
  mirror:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Push docs/ to the vanity repo
        run: |
          git clone --depth 1 \
            https://x-access-token:${{ secrets.PAGES_MIRROR_TOKEN }}@github.com/skimsearchagent/skimsearchagent.github.io.git out
          rm -rf out/* && cp -r docs/. out/
          cd out
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          git diff --quiet --cached || git commit -m "sync project page from skim-search-agent@${GITHUB_SHA::7}"
          git push
```

Until that secret exists, leave this job out — the workflow as shipped only deploys route 1.

## 3. A custom domain

If you would rather own the name (e.g. `skimsearchagent.org`): buy the domain, add
`docs/CNAME` containing the bare domain, and point DNS at GitHub —
four `A` records for the apex (`185.199.108-111.153`) or a `CNAME` for `www` to
`<owner>.github.io`. Then Settings → Pages → Custom domain, and tick *Enforce HTTPS* once the
certificate is issued.

## Checking it before you publish

The page is self-contained, so just open it:

```bash
open docs/index.html                 # or: python3 -m http.server -d docs 8080
```

Everything except the figures is inline, and the figures are relative paths under
`assets/`, so what you see locally is exactly what gets served.

## Note on the live demo

The project page is static and safe to publish anywhere. The **demo** (`demo/server.py`) is
not: it runs a real agent loop with the visitor's own API key and must stay a local,
run-it-yourself tool unless you put a rate-limited, key-custodying backend in front of it.
The project page therefore links to the repository's setup instructions rather than to a
hosted instance.
