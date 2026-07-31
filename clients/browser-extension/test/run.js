// Copyright 2025 SGLang Team
// Licensed under the Apache License, Version 2.0
//
// Runner for the pure-logic tests. No framework and no dependency, so this is
//
//     node test/run.js
//
// from clients/browser-extension/ and nothing else -- there is no npm install
// step and no browser CI harness. The parts of the extension that touch a
// page, a socket or an extension API are not covered here; ../README.md lists
// them as manual steps for Chrome and Firefox.
//
// The same case list also runs under any other JS engine that can eval the
// two files; see test_browser_extension.py for the Python-side contract
// checks that run in CI, where no JS engine is assumed to exist.

"use strict";

const cases = require("./cases.js");

let passed = 0;
let failed = 0;
const failures = [];

function assertions(name) {
  return {
    ok(value, message) {
      if (value) {
        passed++;
      } else {
        failed++;
        failures.push(`${name}: ${message || "expected a truthy value"}`);
      }
    },
    equal(actual, expected, message) {
      if (actual === expected) {
        passed++;
      } else {
        failed++;
        failures.push(
          `${name}: ${message || "values differ"} -- got ${JSON.stringify(
            actual
          )}, expected ${JSON.stringify(expected)}`
        );
      }
    },
    throws(fn, fragment) {
      try {
        fn();
      } catch (err) {
        const text = String(err && err.message ? err.message : err);
        if (!fragment || text.indexOf(fragment) !== -1) {
          passed++;
          return;
        }
        failed++;
        failures.push(`${name}: threw ${text}, expected a message containing ${fragment}`);
        return;
      }
      failed++;
      failures.push(`${name}: expected a throw`);
    },
  };
}

for (const testCase of cases) {
  try {
    testCase.fn(assertions(testCase.name));
  } catch (err) {
    failed++;
    failures.push(`${testCase.name}: unexpected exception ${err && err.stack}`);
  }
}

for (const line of failures) {
  console.error("FAIL " + line);
}
console.log(`${cases.length} cases, ${passed} assertions passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
