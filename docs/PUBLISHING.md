# Publishing the project page

`docs/index.html` is one self-contained file. No build step, no dependencies, plus the figures in
`docs/assets/`. Any static host will serve it. Three routes follow, cheapest first.

## 1. Project page on this repo (2 minutes, no new accounts)

**Do this once**; it requires admin on the repo. Go to Repository →
**Settings → Pages → Build and deployment → Source: `GitHub Actions`**. Then open the Actions tab,
find the failed *Deploy project page* run, and select **Re-run all jobs**.

That step is required under the `ielab` org. The workflow does ask to create the Pages site
itself (`configure-pages` with `enablement: true`), but `GITHUB_TOKEN` is not allowed to create
Pages sites here, so the run fails with *"Create Pages site failed: Resource not accessible by
integration"*. Creating the site by hand is the only way past it. Once it exists, the token's
`pages: write` is enough to deploy, and every later push deploys.

After that, `.github/workflows/pages.yml` redeploys whenever a push to `main` touches `docs/` or
the workflow file itself. It can also be triggered by hand from the Actions tab
(`workflow_dispatch`). The page appears at:

```
https://ielab.github.io/skim-search-agent/
```

To use no workflow at all, choose *Deploy from a branch* → `main` → `/docs` instead. Same result.
The workflow exists so the deploy shows up in the Actions tab and can be re-run.

## 2. The vanity URL, `skimsearchagent.github.io`

GitHub serves `https://<name>.github.io/` from a repository **named exactly** `<name>.github.io`,
owned by a user or organisation **named** `<name>`. This URL therefore needs an account or
organisation called `skimsearchagent`. The `ielab/skim-search-agent` repo alone cannot serve it.

Steps:

1. Check whether the name is free at `https://github.com/skimsearchagent`. A 404 means it is
   available.
2. Create a **free organisation** called `skimsearchagent`
   (github.com/organizations/plan). An organisation is preferable to a personal account: the lab
   can co-own it, and it outlives any one member's account.
3. In that organisation, create a **public** repository named `skimsearchagent.github.io`.
4. Publish the page at that repository's root:

   ```bash
   git clone https://github.com/skimsearchagent/skimsearchagent.github.io
   cd skimsearchagent.github.io
   cp -r /path/to/skim-search-agent/docs/* .      # index.html, assets/, .nojekyll
   git add -A && git commit -m "publish project page" && git push
   ```

5. Settings → Pages → Source: *Deploy from a branch* → `main` → `/ (root)`.

It is live within a minute at `https://skimsearchagent.github.io/`.

### Keeping the mirror in sync automatically

Optional. In the `skimsearchagent` organisation, create a fine-grained personal access token with
*Contents: read and write* on `skimsearchagent.github.io`. Add it to **this** repository as the
secret `PAGES_MIRROR_TOKEN`, then add this job to `.github/workflows/pages.yml`:

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

Leave that job out until the secret exists. As shipped, the workflow only does route 1.

## 3. A custom domain

To own the name (for example `skimsearchagent.org`): buy the domain, add a file named `CNAME` under the docs folder
containing the bare domain, and point DNS at GitHub. That is four `A` records for the apex
(`185.199.108-111.153`), or a `CNAME` for `www` pointing at `<owner>.github.io`. Then go to
Settings → Pages → Custom domain, and tick *Enforce HTTPS* once the certificate is issued.

## Checking it before you publish

The page is self-contained. Open it:

```bash
open docs/index.html                 # or: python3 -m http.server -d docs 8080
```

Everything but the figures is inline, and the figures are relative paths under `assets/`, so the
local view is what gets served.

## Note on the live demo

The project page is static and safe to publish anywhere. The **demo** (`demo/server.py`) is not.
It runs a real agent loop with whatever API key the visitor types in, so it stays a local tool
unless a rate-limited, key-custodying backend is placed in front of it. The project page therefore
links to the repository's setup instructions instead of to a hosted instance.
