# Typos checker

## how to reproduce

1. Install [lookout](https://github.com/src-d/lookout) locally
```shell
$ go get github.com/src-d/lookout
```

2. From lookout directory, let's run the app dependencies using `docker compose`
```shell
$ docker-compose up bblfsh postgres
```
and initialize lookout database
```shell
$ lookoutd migrate
```

3. Run lookout server
```shell
$ lookoutd serve --github-token <gh-token> --github-user <user> <repository>
```
where:
- `user` is the github user handler that will post the comment,
- `gh-token` is [the token of the user](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line) that will post the comment,
- `repository` is the repository url that will be watched for new PRs (it MUST be public)

4. Install style-analyzer as it can be found at [irinakhismatullina/style-analyzer::typos-analyzer](https://github.com/irinakhismatullina/style-analyzer/tree/feature/typos-analyzer)
```shell
$ python3 -m lookout run lookout.style.typos_checker -c config.yml
```

5. Open a PR with a Java file, containing some typos in some code comments
```java
package com.iluwatar.callback;

/**
 * 
 * Callback interface
 * 
 */
// This coment should Be procesed poperly
public interface Callback {
    // i hote java with a pasion tokuns gety weihgt lenhgt
  void call();
}
```
