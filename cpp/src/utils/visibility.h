/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ARCTERN_UTIL_VISIBILITY_H
#define ARCTERN_UTIL_VISIBILITY_H

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(_MSC_VER)
#pragma warning(disable : 4251)
#else
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#ifdef ARCTERN_STATIC
#define ARCTERN_EXPORT
#elif defined(ARCTERN_EXPORTING)
#define ARCTERN_EXPORT __declspec(dllexport)
#else
#define ARCTERN_EXPORT __declspec(dllimport)
#endif

#define ARCTERN_NO_EXPORT
#else  // Not Windows
#ifndef ARCTERN_EXPORT
#define ARCTERN_EXPORT __attribute__((visibility("default")))
#endif
#ifndef ARCTERN_NO_EXPORT
#define ARCTERN_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif  // Non-Windows

// This is a complicated topic, some reading on it:
// http://www.codesynthesis.com/~boris/blog/2010/01/18/dll-export-cxx-templates/
#if defined(_MSC_VER) || defined(__clang__)
#define ARCTERN_TEMPLATE_CLASS_EXPORT
#define ARCTERN_TEMPLATE_EXPORT ARCTERN_EXPORT
#else
#define ARCTERN_TEMPLATE_CLASS_EXPORT ARCTERN_EXPORT
#define ARCTERN_TEMPLATE_EXPORT
#endif

#endif  // ARCTERN_UTIL_VISIBILITY_H
