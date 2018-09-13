# Typos checker

**Typos checker** is a module for **Style Analyzer** for **[lookout](https://github.com/src-d/lookout)**, which points out typos in code comments when they're introduced or modified by a Pull Request.

# How to run it

You can follow these steps to reproduce the demo, as it was seen during the presentation of the hackathon project.

PR and issues are very welcomed ;)

## Run lookout

You will need to have [Go](https://golang.org/doc/install#install) and [docker-compose](https://docs.docker.com/compose/install/) installed in your machine.

### 1. Install [lookout](https://github.com/src-d/lookout)

```shell
$ go get github.com/src-d/lookout/...
```

### 2. Run lookout dependencies using `docker compose`

**lookout** requires a running [bblfsh](https://doc.bblf.sh) server and a [Postgresql](https://www.postgresql.org/) database; to get them you can run from **lookout** directory:

```shell
$ docker-compose up bblfsh postgres
```

Next, initialize **lookout** database

```shell
$ lookoutd migrate
```

### 3. Run lookout server

```shell
$ lookoutd serve --github-token <gh-token> --github-user <user> <repository>
```

where:
- `user` is the github user handler that will post the comment,
- `gh-token` is [the token of the user](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line) that will post the comment,
- `repository` is the repository URL that will be watched for new PRs (it MUST be public)

## Run Style Analyzer

You will need to have [Python](https://www.python.org/downloads/) (between `3.5` and `3.7`)

### 1. Install style-analyzer

Download the **Style Analyzer** repsitory, containing the `typos-checker` module, that can be found at [`irinakhismatullina/style-analyzer` at `typos-analyzer` branch](https://github.com/irinakhismatullina/style-analyzer/tree/feature/typos-analyzer)

Next, install its dependencies, running from its folder:

```shell
$ sudo pip3 install -e .
```

### 2. Configure and run the analyzer

Change the analyzer [`config.yml`](https://github.com/irinakhismatullina/style-analyzer/blob/feature/typos-analyzer/config.yml) to expose the `10302` port.

Next, run it from **Style Analyzer** directory:

```shell
$ python3 -m lookout run lookout.style.typos_checker -c config.yml
```

## Test it with an example in GitHub

Open a PR &ndash;into the watched repository&ndash; with a Java file, containing some typos in some code comments, for example:

`example.java`

```java
package com.iluwatar.callback;

// This coment should Be procesed
public interface Callback {
    // i hote java with a pasion tokuns gety weihgt
  void call();
}
```

Once you have created the PR, and lookout has processed it, you will see a comment for every spotted typo, pointing it out and proposing some alternatives.
