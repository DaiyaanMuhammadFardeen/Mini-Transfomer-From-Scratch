"""
Programming-specific terminology and patterns for enhanced vocabulary building.
This module contains common programming terms, language names, framework names,
and other technical terminology that should be preserved in the vocabulary.
"""
import re
from typing import Dict, List, Set, Tuple


# Programming language names
PROGRAMMING_LANGUAGES = {
    # Most common languages
    'python', 'java', 'javascript', 'typescript', 'c', 'cpp', 'csharp', 'go', 'rust',
    'swift', 'kotlin', 'scala', 'ruby', 'php', 'perl', 'r', 'matlab', 'sql',
    'html', 'css', 'dart', 'lua', 'objective-c', 'shell', 'bash', 'powershell',
    'assembly', 'fortran', 'cobol', 'julia', 'clojure', 'erlang', 'haskell',
    'elixir', 'v', 'zig', 'solidity'
}

# Frameworks and libraries
FRAMEWORKS_LIBRARIES = {
    # Python
    'django', 'flask', 'fastapi', 'requests', 'numpy', 'pandas', 'tensorflow', 'pytorch',
    'scikit-learn', 'keras', 'sqlalchemy', 'celery', 'sanic', 'tornado', 'aiohttp',
    'pyramid', 'bottle', 'web2py', 'falcon', 'starlette', 'pydantic', 'click',
    'pytest', 'unittest', 'coverage', 'django-rest-framework', 'gunicorn', 'uwsgi',
    # JavaScript/Node.js
    'react', 'angular', 'vue', 'express', 'nextjs', 'nuxtjs', 'svelte', 'jquery', 
    'lodash', 'underscore', 'axios', 'redux', 'mobx', 'rxjs', 'webpack', 'babel',
    'jest', 'mocha', 'chai', 'jasmine', 'cypress', 'puppeteer', 'playwright',
    'node', 'npm', 'yarn', 'pnpm', 'vite', 'rollup', 'parcel', 'gatsby', 'remix',
    'apollo', 'graphql', 'react-query', 'swr', 'material-ui', 'ant-design',
    'chakra-ui', 'tailwind', 'bootstrap', 'electron', 'react-native', 'expo',
    'babel', 'typescript', 'eslint', 'prettier', 'webpack', 'rollup', 'vite',
    # Java
    'spring', 'spring-boot', 'hibernate', 'struts', 'jsf', 'vaadin', 'dropwizard',
    'grails', 'micronaut', 'quarkus', 'spring-security', 'spring-data',
    'spring-cloud', 'mybatis', 'junit', 'mockito', 'testng',
    # C#
    'aspnet', 'aspnetcore', 'mvc', 'webapi', 'entity-framework', 'nhibernate',
    'xamarin', 'maui', 'unity', 'nunit', 'xunit', 'moq',
    # Go
    'gin', 'echo', 'beego', 'fiber', 'gorilla', 'chi', 'gofiber', 'golang',
    # Rust
    'actix', 'rocket', 'axum', 'warp', 'tide', 'nickel', 'iron', 'tokio', 'serde',
    # Ruby
    'rails', 'sinatra', 'hanami', 'padrino', 'cuba', 'rack', 'rspec', 'cucumber',
    # PHP
    'laravel', 'symfony', 'codeigniter', 'zend', 'cakephp', 'yii', 'phalcon',
    'silex', 'slim', 'phoenix', 'composer', 'phpunit',
    # Other languages
    'rails', 'sinatra', 'hanami', 'padrino', 'cuba', 'rack', 'rspec', 'cucumber',
    'laravel', 'symfony', 'codeigniter', 'zend', 'cakephp', 'yii', 'phalcon',
    'silex', 'slim', 'phoenix', 'composer', 'phpunit'
}

# Common system commands
SYSTEM_COMMANDS = {
    # File system commands
    'ls', 'cd', 'mkdir', 'rm', 'cp', 'mv', 'pwd', 'cat', 'grep', 'find', 'ps', 'kill',
    'top', 'htop', 'df', 'du', 'chmod', 'chown', 'tar', 'zip', 'unzip', 'ssh', 'scp',
    'rsync', 'ln', 'touch', 'head', 'tail', 'sort', 'uniq', 'wc', 'diff', 'patch',
    'file', 'which', 'whereis', 'locate', 'updatedb', 'md5sum', 'sha256sum', 'dd',
    'mount', 'umount', 'fdisk', 'parted', 'mkfs', 'fsck', 'chattr', 'lsattr',
    # Network commands
    'netstat', 'ss', 'ifconfig', 'ip', 'ping', 'traceroute', 'nslookup', 'dig',
    'host', 'whois', 'arp', 'route', 'iftop', 'nethogs', 'vnstat',
    # Process and system commands
    'lsof', 'iotop', 'iostat', 'vmstat', 'free', 'uptime', 'w', 'who', 'last',
    'lastlog', 'finger', 'users', 'groups', 'id', 'passwd', 'shadow', 'sudo', 'su',
    # Scheduling and services
    'crontab', 'batch', 'atq', 'atrm', 'cron', 'anacron', 'systemd', 'systemctl',
    'service', 'init', 'chkconfig',
    # Development tools
    'git', 'svn', 'hg', 'docker', 'kubectl', 'cmake', 'ninja', 'gcc', 'g++',
    'clang', 'javac', 'python', 'node', 'npm', 'yarn', 'pip', 'conda', 'brew',
    'apt', 'yum', 'dnf', 'pacman', 'systemctl', 'service', 'systemd'
}

# Common technical terms and concepts
TECHNICAL_TERMS = {
    # Architecture patterns
    'microservice', 'monolith', 'soa', 'mvc', 'mvp', 'mvvm', 'singleton', 'factory',
    'observer', 'strategy', 'decorator', 'adapter', 'facade', 'proxy', 'command',
    'state', 'visitor', 'iterator', 'template', 'bridge', 'flyweight', 'composite',
    'builder', 'prototype', 'abstract-factory', 'mediator', 'memento', 'interpreter',
    'chain-of-responsibility', 'active-record', 'data-mapper', 'repository',
    'unit-of-work', 'service-layer', 'cqrs', 'event-sourcing', 'eda', 'soa',
    # Database terms
    'mongodb', 'postgresql', 'mysql', 'sqlite', 'redis', 'cassandra',
    'couchdb', 'elasticsearch', 'influxdb', 'couchbase', 'oracle', 'mssql', 'db2',
    'couchbase', 'riak', 'dynamodb', 'firebase', 'realm', 'neo4j', 'orientdb',
    'arangodb', 'memcached', 'hazelcast', 'etcd', 'consul',
    # DevOps tools
    'jenkins', 'travis', 'circleci', 'github-actions', 'gitlab-ci', 'teamcity',
    'ansible', 'puppet', 'chef', 'terraform', 'vagrant', 'virtualbox', 'vmware',
    'kubernetes', 'openshift', 'docker-compose', 'prometheus', 'grafana', 'datadog',
    'newrelic', 'splunk', 'logstash', 'kibana', 'elk', 'graylog', 'fluentd',
    'sentry', 'rollbar', 'airbrake', 'appsignal', 'bugsnag', 'raygun',
    'consul', 'vault', 'nomad', 'packer', 'artifactory', 'nexus', 'harbor',
    'nagios', 'icinga', 'zabbix', 'munin', 'cacti', 'netdata', 'telegraf',
    # Testing terms
    'unittest', 'integration-test', 'e2e-test', 'tdd', 'bdd', 'mock', 'stub',
    'spy', 'fuzzing', 'mutation', 'coverage', 'selenium', 'cypress', 'playwright',
    'jest', 'mocha', 'chai', 'sinon', 'enzyme', 'testing-library', 'pytest',
    'nose', 'tox', 'behave', 'lettuce', 'robot', 'pytest-bdd', 'cucumber', 'gherkin',
    # Security terms
    'oauth', 'jwt', 'ssl', 'tls', 'aes', 'rsa', 'sha', 'md5', 'cors',
    'csrf', 'xss', 'sql-injection', 'ddos', 'firewall', 'vpn', 'pgp',
    'gpg', 'keycloak', 'saml', 'openid', 'oauth2', 'oidc', 'sso', 'mfa',
    '2fa', 'totp', 'hotp', 'yubikey', 'certbot', 'letsencrypt', 'acme',
    'symantec', 'verisign', 'digicert', 'comodo', 'entrust', 'globalsign',
    # Cloud platforms
    'aws', 'azure', 'gcp', 'heroku', 'vercel', 'netlify', 'digitalocean', 'linode',
    'vultr', 'upcloud', 'hcloud', 'scaleway', 'ovh', 'gandi', 'cloudflare',
    'fastly', 'akamai', 's3', 'ec2', 'lambda', 'dynamodb', 'rds',
    'sqs', 'sns', 'iam', 'vpc', 'eks', 'fargate', 'step-functions', 'api-gateway',
    'app-engine', 'compute-engine', 'kubernetes-engine', 'cloud-run', 'bigquery',
    'firestore', 'firebase', 'cloud-build', 'artifact-registry', 'secret-manager',
    'key-vault', 'app-service', 'function-app', 'cosmos-db', 'azure-devops',
    'azure-pipelines', 'azure-monitor', 'log-analytics', 'application-insights',
    # Data science/ml terms
    'machine-learning', 'deep-learning', 'neural-network', 'cnn', 'rnn', 'lstm',
    'svm', 'random-forest', 'gradient-boosting', 'nlp', 'computer-vision',
    'reinforcement-learning', 'data-pipeline', 'etl', 'data-lake', 'data-warehouse',
    'hadoop', 'spark', 'kafka', 'airflow', 'databricks', 'sagemaker', 'vertex-ai',
    'tensorboard', 'mlflow', 'kubeflow', 'airbyte', 'fivetran', 'matillion',
    'dbt', 'looker', 'tableau', 'power-bi', 'qlik', 'sas', 'r-studio', 'jupyter',
    'colab', 'kaggle', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
    'plotly', 'bokeh', 'altair', 'd3', 'vega', 'vegalite', 'ggplot2',
    # Web technologies
    'ajax', 'xhr', 'fetch', 'websocket', 'sse', 'pwa', 'spa', 'mpa', 'ssr', 'csr', 'isr', 'ssg',
    'bundler', 'transpiler', 'polyfill', 'shim', 'cdn', 'edge-computing',
    'serverless', 'faas', 'baas', 'paas', 'iaas', 'saas', 'bem', 'oocss', 'smacss',
    'atomic-css', 'tachyons', 'tailwind', 'bulma', 'foundation', 'semantic-ui',
    'uikit', 'pure', 'milligram', 'spectre', 'picnic', 'min', '98', 'windows-95',
    # Mobile development
    'flutter', 'react-native', 'xamarin', 'ionic', 'cordova', 'capacitor', 'maui',
    'kotlin-native', 'swiftui', 'compose', 'xcode', 'android-studio', 'gradle',
    'cocoapods', 'carthage', 'fastlane', 'testflight', 'play-store', 'app-store',
    'google-play', 'app-center', 'fabric', 'crashlytics', 'firebase', 'onesignal',
    'urban-airship', 'braze', 'intercom', 'mixpanel', 'amplitude', 'heap',
    'hotjar', 'fullstory', 'logrocket', 'sentry', 'bugsnag', 'raygun',
    # Performance and monitoring
    'profiling', 'benchmarking', 'load-testing', 'stress-testing', 'apdex',
    'response-time', 'throughput', 'latency', 'bandwidth', 'concurrency',
    'parallelism', 'thread', 'process', 'mutex', 'semaphore',
    'lock', 'atomic', 'promise', 'callback', 'closure',
    'lambda', 'generator', 'iterator', 'coroutine', 'fiber', 'greenlet',
    'event-loop', 'non-blocking', 'blocking', 'io', 'cpu-bound', 'io-bound',
    'thread-pool', 'connection-pool', 'object-pool', 'memory-pool', 'gc',
    'garbage-collection', 'memory-leak', 'stack-overflow', 'heap', 'stack',
    'heap-overflow', 'buffer-overflow', 'race-condition', 'deadlock', 'livelock',
    'starvation', 'priority-inversion', 'context-switch', 'preemption', 'quantum',
    'round-robin', 'fifo', 'lifo', 'priority-scheduling', 'multithreading',
    'multiprocessing', 'multiprogramming', 'multitasking', 'timesharing',
    'real-time', 'soft-real-time', 'hard-real-time', 'embedded', 'firmware',
    'bare-metal', 'rtos', 'freertos', 'zephyr', 'riot', 'mynewt', 'contiki',
    'nutttx', 'embos', 'threadx', 'ucos', 'vxworks', 'qnx', 'integrity',
    'palm', 'symbian', 'blackberry', 'webos', 'tizen', 'android', 'ios',
    'watchos', 'tvos', 'carplay', 'android-auto', 'wearos', 'tizen',
    # Version control and collaboration
    'git', 'svn', 'hg', 'fossil', 'bazaar', 'cvs', 'perforce', 'clearcase',
    'team-foundation-server', 'tfs', 'visual-studio-team-services', 'vsts',
    'azure-devops', 'github', 'gitlab', 'bitbucket', 'sourceforge', 'codebase',
    'gitea', 'gogs', 'phabricator', 'fogbugz', 'jira', 'confluence', 'trello',
    'asana', 'monday', 'clickup', 'notion', 'linear', 'shortcut', 'pivotal',
    'pivotal-tracker', 'waffle', 'zenhub', 'huboard', 'agile', 'scrum', 'kanban',
    'sprint', 'backlog', 'user-story', 'epic', 'task', 'bug', 'feature',
    'story-point', 'velocity', 'burndown', 'retrospective', 'standup', 'planning',
    'review', 'merge', 'pull-request', 'merge-request', 'branch', 'tag', 'commit',
    'checkout', 'clone', 'fetch', 'pull', 'push', 'rebase', 'merge', 'cherry-pick',
    'stash', 'reset', 'revert', 'blame', 'log', 'diff', 'status', 'add', 'rm',
    'mv', 'config', 'remote', 'origin', 'upstream', 'fork', 'upstream', 'origin',
    'master', 'main', 'develop', 'feature', 'release', 'hotfix', 'bugfix',
    'wip', 'work-in-progress', 'draft', 'ready-for-review', 'rfc', 'request-for-comments',
    'pr', 'mr', 'pull-request', 'merge-request', 'issue', 'ticket', 'story',
    'commit-message', 'git-hook', 'pre-commit', 'post-commit', 'pre-push',
    'post-receive', 'git-flow', 'github-flow', 'gitlab-flow', 'trunk-based',
    'continuous-integration', 'ci', 'continuous-deployment', 'cd', 'continuous-delivery',
    'devops', 'gitops', 'mlops', 'aioops', 'sre', 'site-reliability-engineering',
    'infrastructure-as-code', 'iac', 'configuration-management', 'provisioning',
    'orchestration', 'containerization', 'virtualization', 'cloud-native',
    'microservices', 'service-mesh', 'api-gateway', 'load-balancer', 'reverse-proxy',
    'cdn', 'edge-computing', 'serverless', 'faas', 'baas', 'paas', 'iaas', 'saas'
}

# Common programming constructs and keywords (language-agnostic)
PROGRAMMING_CONSTRUCTS = {
    'function', 'class', 'method', 'variable', 'constant', 'array', 'list', 'map',
    'dictionary', 'object', 'interface', 'enum', 'struct', 'union', 'typedef',
    'import', 'export', 'namespace', 'package',
    'module', 'library', 'framework', 'api', 'endpoint', 'route', 'url', 'uri',
    'json', 'xml', 'yaml', 'toml', 'ini', 'config', 'settings', 'environment',
    'debug', 'release', 'production', 'staging', 'development', 'test', 'ci', 'cd',
    'pipeline', 'workflow', 'build', 'deploy', 'release', 'version', 'commit',
    'branch', 'merge', 'pull-request', 'issue', 'bug', 'feature', 'refactor',
    'optimization', 'performance', 'cache', 'buffer', 'memory', 'cpu', 'gpu',
    'thread', 'process', 'mutex', 'semaphore', 'lock', 'atomic', 'async', 'await',
    'promise', 'callback', 'closure', 'lambda', 'generator', 'iterator',
    'exception', 'error', 'warning', 'log', 'trace', 'debug', 'info', 'error',
    'def', 'class', 'var', 'let', 'const',
    'public', 'private', 'protected', 'static', 'final', 'abstract', 'virtual',
    'override', 'interface', 'extends', 'implements', 'super', 'this', 'self',
    'sizeof', 'typeof', 'instanceof', 'null',
    'undefined', 'true', 'false', 'nil', 'void', 'int', 'float', 'double',
    'string', 'bool', 'char', 'byte', 'short', 'long', 'unsigned', 'signed',
    'volatile', 'transient', 'synchronized', 'native', 'strictfp', 'sealed',
    'record', 'union', 'typedef', 'template', 'generic', 'trait',
    'mixin', 'protocol', 'extension', 'operator', 'constructor', 'destructor',
    'function-pointer', 'higher-order-function', 'first-class-function',
    'pure-function', 'impure-function', 'side-effect', 'referential-transparency',
    'imperative', 'declarative', 'functional', 'object-oriented', 'procedural',
    'logic', 'event-driven', 'reactive', 'concurrent', 'parallel', 'distributed',
    'synchronous', 'asynchronous', 'blocking', 'non-blocking', 'cooperative',
    'preemptive', 'coroutine', 'async', 'await', 'future', 'promise',
    'observable', 'stream', 'reactive', 'functional-reactive', 'monad', 'functor',
    'applicative', 'category-theory', 'type-theory', 'dependent-type', 'polymorphism',
    'parametric-polymorphism', 'ad-hoc-polymorphism', 'subtype-polymorphism',
    'parametric', 'bounded', 'unbounded', 'covariance', 'contravariance',
    'invariance', 'generic', 'template', 'monomorphization', 'specialization',
    'overloading', 'overriding', 'virtual-function', 'pure-virtual', 'abstract',
    'interface', 'trait', 'mixin', 'duck-typing', 'gradual-typing', 'static-typing',
    'dynamic-typing', 'strong-typing', 'weak-typing', 'manifest-typing', 'latent',
    'nominal-typing', 'structural-typing', 'row-polymorphism', 'bounded-quantification',
    'higher-ranked-types', 'higher-kinded-types', 'type-class', 'implicit-parameter',
    'type-constraint', 'type-bound', 'type-parameter', 'type-variable', 'type-alias',
    'type-definition', 'type-inference', 'type-checking', 'type-safety', 'soundness',
    'completeness', 'decidability', 'type-erasure', 'type-coercion', 'type-casting',
    'upcast', 'downcast', 'widening', 'narrowing', 'boxing', 'unboxing',
    'autoboxing', 'manual-boxing', 'heap-allocation', 'stack-allocation',
    'garbage-collection', 'reference-counting', 'tracing-gc', 'mark-sweep',
    'copying-gc', 'generational-gc', 'incremental-gc', 'concurrent-gc', 'real-time-gc',
    'memory-management', 'memory-leak', 'dangling-pointer', 'buffer-overflow',
    'use-after-free', 'double-free', 'memory-corruption', 'memory-layout',
    'data-structure', 'algorithm', 'complexity', 'big-o', 'time-complexity',
    'space-complexity', 'amortized', 'worst-case', 'best-case', 'average-case',
    'recursion', 'iteration', 'tail-recursion', 'mutual-recursion', 'tree-recursion',
    'linear-recursion', 'binary-recursion', 'divide-and-conquer', 'dynamic-programming',
    'greedy', 'backtracking', 'branch-and-bound', 'brute-force', 'heuristic',
    'approximation', 'randomized', 'parallel', 'distributed', 'online', 'offline',
    'exact', 'approximate', 'deterministic', 'non-deterministic', 'probabilistic',
    'quantum', 'quantum-computing', 'quantum-algorithm', 'shors-algorithm',
    'grovers-algorithm', 'quantum-fourier-transform', 'quantum-circuit',
    'quantum-gate', 'qubit', 'superposition', 'entanglement', 'quantum-noise',
    'quantum-error-correction', 'quantum-communication', 'quantum-cryptography',
    'post-quantum', 'quantum-safe', 'quantum-resistant', 'crypto-agility'
}

# HTTP methods and status codes
HTTP_TERMS = {
    'get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'connect', 'trace',
    '200', '201', '202', '204', '301', '302', '304', '400', '401', '403', '404',
    '405', '500', '502', '503', '504', 'ok', 'created', 'accepted', 'no-content',
    'not-found', 'internal-server-error', 'bad-request', 'unauthorized', 'forbidden',
    'redirect', 'permanent-redirect', 'temporary-redirect', 'moved-permanently',
    'see-other', 'not-modified', 'temporary-redirect', 'permanent-redirect',
    'bad-gateway', 'service-unavailable', 'gateway-timeout', 'unprocessable-entity',
    'too-many-requests', 'request-timeout', 'conflict', 'gone', 'precondition-failed',
    'payload-too-large', 'uri-too-long', 'unsupported-media-type', 'range-not-satisfiable',
    'expectation-failed', 'misdirected-request', 'unprocessable-entity', 'locked',
    'failed-dependency', 'upgrade-required', 'precondition-required', 'too-many-requests',
    'request-header-fields-too-large', 'unavailable-for-legal-reasons', 'internal-server-error',
    'not-implemented', 'bad-gateway', 'service-unavailable', 'gateway-timeout',
    'http', 'https', 'url', 'uri', 'endpoint', 'rest', 'graphql', 'soap',
    'json', 'xml', 'yaml', 'content-type', 'accept', 'authorization', 'bearer',
    'oauth', 'jwt', 'cookie', 'session', 'cors', 'csrf', 'xss', 'samesite',
    'secure', 'httponly', 'cache-control', 'etag', 'last-modified', 'if-modified-since',
    'if-none-match', 'vary','content-length', 'content-encoding',
    'content-language', 'content-location', 'content-disposition', 'age',
    'expires', 'date', 'location', 'retry-after', 'server', 'www-authenticate',
    'proxy-authenticate', 'proxy-authorization', 'te', 'trailer', 'transfer-encoding',
    'upgrade', 'via', 'warning', 'dnt', 'x-forwarded-for', 'x-forwarded-host',
    'x-forwarded-proto', 'x-real-ip', 'x-request-id', 'x-correlation-id',
    'x-powered-by', 'x-content-type-options', 'x-frame-options', 'x-xss-protection',
    'strict-transport-security', 'content-security-policy', 'x-permitted-cross-domain-policies',
    'referrer-policy', 'feature-policy', 'permissions-policy', 'access-control-allow-origin',
    'access-control-allow-credentials', 'access-control-allow-headers',
    'access-control-allow-methods', 'access-control-expose-headers',
    'access-control-max-age', 'access-control-request-headers',
    'access-control-request-method', 'origin', 'timing-allow-origin'
}

# File extensions and types
FILE_TYPES = {
    # Code files
    'py', 'js', 'ts', 'java', 'cpp', 'c', 'cs', 'go', 'rs', 'swift', 'kt', 'scala',
    'rb', 'php', 'html', 'css', 'json', 'xml', 'yaml', 'yml', 'toml', 'ini', 'cfg',
    'txt', 'md', 'rst', 'csv', 'tsv', 'sql', 'sh', 'bash', 'zsh', 'bat', 'cmd',
    'dockerfile', 'makefile', 'cmakelists', 'gradle', 'sbt', 'cabal', 'toml',
    'lock', 'env', 'gitignore', 'dockerignore', 'npmignore', 'editorconfig',
    'gitattributes', 'prettierrc', 'eslintignore', 'eslintrc', 'babelrc',
    'webpack', 'jest', 'babel', 'ts', 'tsx', 'jsx', 'vue', 'svelte', 'astro',
    'graphql', 'gql', 'proto', 'thrift', 'avro', 'capnp', 'flatbuffers',
    'json', 'bson', 'msgpack', 'cbor', 'yaml', 'toml', 'ini', 'cfg', 'conf',
    'properties', 'env', 'config', 'settings', 'prefs', 'ini', 'cfg', 'conf',
    # Data files
    'csv', 'tsv', 'json', 'xml', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf',
    'properties', 'env', 'config', 'settings', 'prefs', 'ini', 'cfg', 'conf',
    'parquet', 'hdf5', 'pkl', 'pickle', 'joblib', 'npz', 'npy', 'h5', 'hdf',
    'db', 'sqlite', 'sqlite3', 'mdb', 'accdb', 'sql', 'sqlitedb', 'db3',
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'svg', 'webp', 'ico', 'psd',
    'ai', 'eps', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'odt',
    'ods', 'odp', 'rtf', 'tex', 'log', 'bak', 'tmp', 'temp', 'zip', 'tar',
    'gz', 'rar', '7z', 'bz2', 'xz', 'iso', 'dmg', 'exe', 'msi', 'deb', 'rpm',
    'apk', 'ipa', 'jar', 'war', 'ear', 'class', 'so', 'dll', 'dylib', 'a',
    'o', 'obj', 'lib', 'out', 'bin', 'elf', 'img', 'vmdk', 'vhd', 'vdi',
    'bak', 'tmp', 'temp', 'swp', 'swo', 'cache', 'log', 'err', 'out', 'pid',
    'sock', 'lock', 'idx', 'dat', 'data', 'txt', 'text', 'md', 'markdown',
    'rst', 'asciidoc', 'adoc', 'org', 'tex', 'bib', 'bst', 'sty', 'cls'
}

# All programming-related terms combined
ALL_PROGRAMMING_TERMS = (
    PROGRAMMING_LANGUAGES | FRAMEWORKS_LIBRARIES | SYSTEM_COMMANDS | 
    TECHNICAL_TERMS | PROGRAMMING_CONSTRUCTS | HTTP_TERMS | FILE_TYPES
)

# Create regex patterns for matching programming terms
def create_programming_patterns() -> Dict[str, str]:
    """Create regex patterns for matching programming terms."""
    patterns = {}
    
    # Programming languages - match as whole words
    lang_pattern = r'\b(' + '|'.join(sorted(PROGRAMMING_LANGUAGES, key=len, reverse=True)) + r')\b'
    patterns['programming_languages'] = lang_pattern
    
    # Frameworks and libraries - match as whole words or with common prefixes/suffixes
    framework_pattern = r'\b(' + '|'.join(sorted(FRAMEWORKS_LIBRARIES, key=len, reverse=True)) + r')\b'
    patterns['frameworks'] = framework_pattern
    
    # System commands - match as whole words
    command_pattern = r'\b(' + '|'.join(sorted(SYSTEM_COMMANDS, key=len, reverse=True)) + r')\b'
    patterns['system_commands'] = command_pattern
    
    # Technical terms - match as whole words
    tech_pattern = r'\b(' + '|'.join(sorted(TECHNICAL_TERMS, key=len, reverse=True)) + r')\b'
    patterns['technical_terms'] = tech_pattern
    
    # Programming constructs - match as whole words
    construct_pattern = r'\b(' + '|'.join(sorted(PROGRAMMING_CONSTRUCTS, key=len, reverse=True)) + r')\b'
    patterns['programming_constructs'] = construct_pattern
    
    # HTTP terms - match as whole words or numbers
    http_pattern = r'\b(' + '|'.join(sorted(HTTP_TERMS, key=len, reverse=True)) + r')\b'
    patterns['http_terms'] = http_pattern
    
    # File types - match with dot prefix
    file_pattern = r'\.(' + '|'.join(sorted(FILE_TYPES, key=len, reverse=True)) + r')\b'
    patterns['file_types'] = file_pattern
    
    return patterns

# Create comprehensive patterns
PROGRAMMING_PATTERNS = create_programming_patterns()


def extract_programming_terms(text: str) -> List[str]:
    """Extract programming-related terms from text using regex patterns."""
    terms = []
    
    # Normalize text for matching
    normalized_text = text.lower()
    
    # Match each pattern category
    for category, pattern in PROGRAMMING_PATTERNS.items():
        matches = re.findall(pattern, normalized_text, re.IGNORECASE)
        for match in matches:
            # Convert match to the canonical form from our sets
            if isinstance(match, tuple):
                match = [m for m in match if m][0] if any(match) else None
            if match:
                match_lower = match.lower()
                # Find the canonical form from our sets
                canonical = None
                if category == 'programming_languages':
                    canonical = next((lang for lang in PROGRAMMING_LANGUAGES if lang.lower() == match_lower), None)
                elif category == 'frameworks':
                    canonical = next((fw for fw in FRAMEWORKS_LIBRARIES if fw.lower() == match_lower), None)
                elif category == 'system_commands':
                    canonical = next((cmd for cmd in SYSTEM_COMMANDS if cmd.lower() == match_lower), None)
                elif category == 'technical_terms':
                    canonical = next((term for term in TECHNICAL_TERMS if term.lower() == match_lower), None)
                elif category == 'programming_constructs':
                    canonical = next((pc for pc in PROGRAMMING_CONSTRUCTS if pc.lower() == match_lower), None)
                elif category == 'http_terms':
                    canonical = next((http for http in HTTP_TERMS if http.lower() == match_lower), None)
                elif category == 'file_types':
                    # For file types, the match might be ".ext", so we check without the dot
                    match_clean = match_lower.lstrip('.')
                    canonical = next((ft for ft in FILE_TYPES if ft.lower() == match_clean), None)
                    if canonical:
                        canonical = f".{canonical}"  # Add back the dot for file types
                
                if canonical and canonical not in terms:
                    terms.append(canonical)
    
    return terms


def create_programming_tokens(terms: List[str]) -> List[str]:
    """Convert programming terms to special tokens."""
    tokens = []
    for term in terms:
        # Create special token format: <LANG_NAME> or <FRAMEWORK_NAME> etc.
        if term in PROGRAMMING_LANGUAGES:
            token = f"<{term.upper()}_LANG>"
        elif term in FRAMEWORKS_LIBRARIES:
            token = f"<{term.upper()}_FRAMEWORK>"
        elif term in SYSTEM_COMMANDS:
            token = f"<{term.upper()}_CMD>"
        elif term in TECHNICAL_TERMS:
            token = f"<{term.upper()}_TERM>"
        elif term in PROGRAMMING_CONSTRUCTS:
            token = f"<{term.upper()}_CONSTRUCT>"
        elif term in HTTP_TERMS:
            token = f"<{term.upper()}_HTTP>"
        elif term.startswith('.'):
            token = f"<{term[1:].upper()}_EXT>"
        else:
            token = f"<{term.upper()}_TECH>"
        
        tokens.append(token)
    
    return tokens


# Test the functionality
if __name__ == "__main__":
    test_text = """
    Added support for Python, JavaScript, and React in the new API.
    Used Docker and Kubernetes for deployment with CI/CD pipeline.
    Implemented JWT authentication with HTTPS endpoints.
    Fixed bug in the SQL query for the PostgreSQL database.
    Used async/await pattern with Promises in the Node.js backend.
    Added .py, .js, .ts, and .md file support.
    """
    
    terms = extract_programming_terms(test_text)
    print("Extracted terms:", terms)
    
    tokens = create_programming_tokens(terms)
    print("Generated tokens:", tokens)