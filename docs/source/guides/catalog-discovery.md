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
Frontmatter accepts LF, CRLF and CR line endings. An opening frontmatter marker
without a closing marker is invalid metadata under each format.
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

The repository includes task-focused declarations for more than the Echo smoke
test. They make these source definitions easier to select without claiming
measured training quality or runtime validation:

| Task family | Source definition | Useful declared signal |
|-------------|-------------------|------------------------|
| Python execution | `envs/coding_env` | Standard output, standard error and exit status |
| Browser interaction | `envs/browsergym_env` | Page observations and browser actions |
| Calendar scheduling | `envs/calendar_env` | Seeded scenarios and declared MCP event tools |
| Chess | `envs/chess_env` | Legal UCI moves, FEN positions and configurable opponents |
| Reasoning questions | `envs/reasoning_gym_env` | Single-step episodes and dataset-defined answer scoring |

For example, after building a snapshot, try
`openenv discover "Python snippets and standard error" --catalog catalog.json`.
These examples are a small authored subset of the complete repository inventory,
not a claim that every source has equally detailed discovery metadata. Calendar's
tool names cite the existing event-tool definitions; interfaces for other sources
are not inferred from their orchestration methods.

Leave `representative_queries` empty or supply two to five hints. The declaration,
generated entry and packaged JSON schemas enforce the same bounds.

Description precedence is `discovery.json`, `openenv.yaml`, package description,
then README frontmatter. An explicit reviewed license declaration takes
precedence; conflicting package and README declarations remain `unknown` with a
diagnostic. Otherwise the package or README declaration is used, then the
repository's declared source license. Mappings are retained in
`metadata.provenance`. This is a source declaration, not a legal certification.

A package `license = {file = "LICENSE"}` table remains `unknown`: a file pointer
alone does not identify an SPDX or custom license. The producer does not classify
the referenced file's contents or replace this unknown with a repository-wide
license. A license table cannot contain both `file` and `text`.

Custom license text is retained internally when comparing package and README
declarations; the emitted card still uses `other`. Two different custom texts
must not match merely because both normalize to that category. Identical custom
text, allowing for line endings and outer whitespace, is a consistent declaration.
Two bare `other` markers do not establish a shared license identity. Ambiguous
or conflicting declarations produce `unknown` and a `license_conflict` warning,
without borrowing a positive license claim from the repository.

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

### Resource type and schema scope

The resource media type is
`application/vnd.openenv.environment-card+json`. It describes an environment
source definition, not an installable MCP-server configuration. Clients dispatch
on the explicit ARD entry `type` and check the card's `data.schema_version`.
Neither a display name nor a version string alone identifies this resource type.

The ARD entry carries search-facing fields such as `displayName`, `description`,
`tags`, `capabilities` and `representativeQueries`. Its inline `data` is the
Environment Card. Preserve both the entry and the complete card rather than
reconstructing the latter from search snippets.

| Packaged schema | Applies to |
|-----------------|------------|
| `environment-card.schema.json` | One `entry.data` Environment Card |
| `catalog.schema.json` | A complete repository snapshot; its `DiscoveryEntry` definition describes each entry |
| `declaration.schema.json` | Producer-side `discovery.json` input, not a discovered resource |

The schema `$id` is a schema identifier, not an instruction to fetch it or a
guarantee that a draft has been published at that location. Pin the agreed
schema/profile revision during review. Loading these self-contained schemas
does not require fetching candidate resource URLs.

### Consumer validation

JSON Schema validation is necessary but is not the whole profile contract.
The packaged schemas enforce object shape, required fields, relative-path
safety, supported literals, conditional artifact/license-evidence presence, and
exactly one orchestration interface. Other rules require semantic validation:

| Subject | Additional rule |
|---------|-----------------|
| Repository source | Parse a credential-free GitHub HTTPS Git URI with no query, fragment or encoded path components; its repository identity must equal `source.id` |
| Artifact binding | Every artifact's `uri`, `path` and `revision` must equal the selected `data.source` tuple |
| Agent-tool declaration | `source_revision` must equal the environment revision; agent-tool protocols must be unique |
| License | Parse an SPDX expression or the explicit `other`/`unknown` sentinel; evidence URLs must be credential-free HTTPS references |
| Framework requirement | Parse the declared package requirement and verify that its normalized package name is `openenv`; this is not runtime compatibility evidence |
| Entry identity | Require the canonical `urn:air:` prefix and the selected full source revision suffix; within a snapshot, the publisher must also match |
| Capabilities | Require a source-bound agent-tool declaration; known simulation-control names are forbidden regardless of case |
| Snapshot | Check source/publisher consistency, unique identities and paths, direct-child inventory scope, complete accounting and the snapshot digest before trusting a refresh |

The Python models enforce these semantic rules. Consumers in other languages
must implement equivalent checks; passing JSON Schema alone must not be labeled
full conformance. A malformed or unsupported card must not become a resolved,
validated or installable environment through guessed defaults.

Tool evidence is still a declaration. Recognizing a protocol or checking a tool
name does not prove its behavior, safety or suitability for training.

### Source retrieval and environment setup

Discovery reads metadata only. Source acquisition, installation and execution
are separate actions:

1. Select and validate a complete card, preserving its full identifier and
   snapshot provenance.
2. Only `artifact_availability: resolvable` supplies an immutable artifact
   locator in this profile. `external` and `unknown` do not authorize a guessed
   download from source metadata.
3. After the caller's policy permits source retrieval, use the Git artifact's
   `uri`, full `revision` and repository-relative `path`. Select that environment
   within the pinned checkout while retaining any required monorepo build
   context. The URN is not a URL, and the GitHub repository ID is not a Hub Space
   identifier.
4. Review the selected environment's setup instructions at the same revision.
   Build/run and provider requirements are environment-specific. Do not assume
   that installing the repository root installs the selected environment.

The card does not specify a universal installer, an OCI image digest, a complete
dependency lock, launch arguments, resource budgets or execution permissions.
`framework_requirement` alone is not an installation recipe. A declared MCP
agent-tool interface is not enough to construct an MCP-server install action.
Automated install/launch handling needs its own explicit producer-consumer
contract; it must not be inferred from discovery metadata.

The first client action is read-only inspection and source navigation. A
`resolvable` card does not establish caller access, a running deployment,
reproducible build output, validated interfaces or approval to execute.

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
