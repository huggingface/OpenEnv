# Discover environments before running them

`openenv catalog` and `openenv discover` implement RFC 011's first repository
profile. They read committed metadata, not candidate Python modules, images or
running deployments. This is separate from `AutoEnv`, which resolves and loads a
known environment.

## Produce a versioned inventory

From a clone of the public OpenEnv repository:

```bash
openenv catalog build \
  --repository . \
  --repository-uri https://github.com/huggingface/OpenEnv.git \
  --revision HEAD \
  --publisher example.org \
  --output catalog.json
```

`example.org` is only an illustrative publication authority. A real publication
must explicitly configure an authority its publisher can substantiate. Supplying
the option is not identity verification, GitHub endorsement, or permission to run
the listed code.

The initial inventory is precisely the regular tracked `openenv.yaml` files in
direct child directories under `envs/` at the resolved Git commit. It excludes
untracked files, submodules, deployments, and community repositories not in that
tree. `--root` changes that explicitly scoped directory. It is not a global
environment census.

The producer reads the commit's manifest, project metadata and README frontmatter.
It keeps a source URI, environment path and full revision on each card and its
Git artifact. The snapshot has a separate content digest. Builds from identical
committed inputs and publisher settings are byte-identical.

An unreadable or invalid eligible environment produces an error in the build
report and `complete: false`; the command exits nonzero. The partial report is
inspectable but cannot be loaded as a complete catalog or used to infer
withdrawals. Unknown license or omitted tool evidence remains unknown.

## Find and inspect a complete record

```bash
openenv discover "client smoke test" --catalog catalog.json
openenv discover "client smoke test" --catalog catalog.json --json
openenv discover "" --catalog catalog.json --filter license=BSD-3-Clause
openenv catalog inspect "<identifier from the result>" --catalog catalog.json
```

The first baseline ranks literal, case-insensitive query-token overlap over
authored descriptions, names, tags, capabilities and representative query hints.
Scores describe this lexical match only. They are not measured training quality,
validation, reputation or execution approval. Exact filters support `license`,
`provider`, `artifact_availability`, `name`, `tags` and `type`; unsupported filters
are errors rather than empty success.

An identifier is resolved only in the explicitly configured snapshot. Missing
identifiers fail explicitly. No URL is guessed from an identifier. Neither the
CLI nor the library fetches metadata URLs, visits deployments or passes source
IDs into a Hub resolver.

## Add only genuinely missing declarations

Optional environment-local `discovery.json` files can add reviewed task
descriptions, tags, representative queries, artifact availability, applicable
license declarations, or declared agent tools. Echo contains a worked example.
The producer reads it from the same Git revision as the environment.

Description precedence is `discovery.json`, `openenv.yaml`, package description,
then README frontmatter. An explicit reviewed license declaration takes
precedence; conflicting package and README declarations remain `unknown` with a
diagnostic. Otherwise the package or README declaration is used, then the
repository's declared source license. Mappings are retained in
`metadata.provenance`. This is a source declaration, not a legal certification.

Tool declarations name repository-relative evidence within the environment and
remain `declared`. Merely listing a tool name does not establish semantic safety.
Simulation controls must not be exposed as agent tools. This profile rejects
known control names as diagnostics but does not claim that name checks prove the
boundary. Discovery never invokes a tool to inspect it.

`manifest_spec_version` is a manifest marker. `framework_requirement` is a
source-declared package requirement. Neither is a runtime-protocol version or a
compatibility certificate.

## Profile and publication contract

The `0.1-draft` profile is declaration-only, GitHub-source, revision-bound, and
inline. Unsupported profile versions and validated-interface claims are
explicitly rejected. It does not replace RFC 008's normalized validation
manifest, report contract, or graders. The normalized validation manifest is not
a prerequisite because ordinary environments may not declare its validation
block.

Schemas are packaged in `openenv.discovery/schemas/0.1-draft/`. Regenerate them
with `PYTHONPATH=src python scripts/generate_discovery_schemas.py`; `--check`
detects drift. The Pydantic contract additionally enforces relational invariants
such as matching artifact revisions and inventory accounting.

The identifier is publisher-scoped and revision-qualified. Its locator component
is SHA-256 over the declared repository URI, a newline and the environment path.
It is not a global canonical identity. A publisher transfer requires an explicit
mapping rather than merging similarly named environments.

Publish the complete generated JSON through an owner-controlled, versioned
metadata channel. Consumers pin the profile they understand and inspect the
whole `entries[].data` card. A third-party catalog adapter may extract entries
but must retain source/path/revision and the snapshot reference. Publication
alone does not guarantee admission to or indexing by any finder.

The `Discovery catalog` workflow generates a revision-named GitHub Actions
artifact from the checked-out source. It uses the hosting GitHub domain and
repository-owner namespace without claiming verified publisher status. The
artifact is a reviewable publication output, not a new publicly hosted registry;
an operator explicitly downloads and configures it in a consumer. On PRs, it
identifies the checked-out PR merge revision rather than pretending to describe
released code.

The local library function `compare_catalogs(previous, current)` distinguishes
added listings, withdrawn listings, superseded revision cards and corrected
metadata for the same revision. Both inputs must be complete snapshots of the
same publisher, source and inventory scope. Failed reads cannot authorize
removal. Historical snapshots remain usable as explicit historical metadata.

For an independent consumer, configure the same snapshot in a compatible finder.
Do not duplicate its domain rules by importing environment runtime code. The
first-source milestone is useful on its own; additional providers, URL fetching,
private credentials and runtime validation require their own supported profiles.
