#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "beautifulsoup4",
#   "html2text",
#   "requests",
# ]
# ///
"""
Unified CUDA documentation scraper.

Scrapes NVIDIA CUDA documentation (PTX ISA, Runtime API, Driver API)
and converts to searchable markdown format.
"""

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin

import html2text
import requests
from bs4 import BeautifulSoup, Tag


class DocumentationScraper:
    """Base class for CUDA documentation scrapers."""

    def __init__(
        self,
        base_url: str,
        output_dir: Path,
        cache_dir: Path | None = None,
        skip_download: bool = False,
        force: bool = False,
    ):
        self.base_url = base_url
        self.output_dir = output_dir
        self.cache_dir = cache_dir or (output_dir.parent / f"{output_dir.name}-raw")
        self.skip_download = skip_download
        self.force = force

        # HTTP session with headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
        )

        # html2text configuration
        self.h2t = html2text.HTML2Text()
        self.h2t.body_width = 0
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.ignore_emphasis = False
        self.h2t.skip_internal_links = False
        self.h2t.unicode_snob = True
        self.h2t.decode_errors = "ignore"

    def fetch_page(self, url: str) -> BeautifulSoup | None:
        """Fetch and parse a webpage."""
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def sanitize_filename(self, name: str, section_num: str = "") -> str:
        """Convert title to safe filename."""
        # Remove section number from title if present
        name = re.sub(r"^\d+(\.\d+)*\.?\s*", "", name)
        name = re.sub(r"#.*$", "", name)  # Remove anchors
        name = re.sub(r"\.html?$", "", name)  # Remove extensions
        name = re.sub(r"[^\w\s\-_.]", "", name)  # Remove special chars
        name = re.sub(r"\s+", "-", name)  # Spaces to hyphens
        name = name.lower().strip("-")

        # Add section number prefix if provided
        if section_num:
            name = f"{section_num}-{name}"

        return name if name else "index"

    def extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main documentation content from page."""
        content = soup.find("div", class_="contents")
        if not content:
            content = (
                soup.find("div", id="doc-content")
                or soup.find(attrs={"role": "main"})
                or soup.find("body")
            )
        if not content:
            raise ValueError("Could not find main content")

        # Remove navigation elements
        for nav in content.find_all(
            ["div", "ul"],
            class_=["header", "headertitle", "navigate", "breadcrumb"],
        ):
            nav.decompose()

        for elem in content.find_all(
            ["div"],
            id=["top", "titlearea", "projectlogo", "projectname", "projectbrief"],
        ):
            elem.decompose()

        # Remove large navigation lists
        for textblock in content.find_all("div", class_="textblock"):
            links = textblock.find_all("a", href=True)
            if len(links) > 10:
                html_links = [
                    link for link in links if link.get("href", "").endswith(".html")
                ]
                if len(html_links) > 10:
                    textblock.decompose()

        return content

    def convert_to_markdown(self, soup: BeautifulSoup, page_url: str) -> str:
        """Convert HTML to markdown."""
        content = self.extract_main_content(soup)

        # Make image URLs absolute
        for img in content.find_all("img"):
            src = img.get("src")
            if src and not src.startswith(("http://", "https://")):
                img["src"] = urljoin(page_url, src)

        # Make link URLs absolute
        for link in content.find_all("a"):
            href = link.get("href")
            if href and not href.startswith(("http://", "https://", "#", "mailto:")):
                link["href"] = urljoin(page_url, href)

        markdown = self.h2t.handle(str(content))
        markdown = self._clean_navigation_markdown(markdown)
        markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
        return markdown.strip()

    def _clean_navigation_markdown(self, markdown: str) -> str:
        """Remove navigation cruft from markdown."""
        lines = markdown.split("\n")
        cleaned_lines = []
        in_nav = False
        found_header = False

        for line in lines:
            if (
                "NVIDIA" in line
                and "Toolkit Documentation" in line
                and not found_header
            ):
                in_nav = True
                continue

            if line.startswith("###") or (
                line.startswith("##") and "Public Members" in line
            ):
                in_nav = False
                found_header = True

            if not in_nav:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)


class APIScraper(DocumentationScraper):
    """Scraper for CUDA Runtime, Driver, and Math API documentation."""

    # Per-type config: (base_url, modules_discovery_path, modules_pattern,
    #                   structs_discovery_path, structs_pattern)
    _CONFIG: dict[str, tuple[str, str, str, str, str]] = {
        "runtime": (
            "https://docs.nvidia.com/cuda/cuda-runtime-api/",
            "modules.html",
            r"group__CUDART.*\.html",
            "annotated.html",
            r"(struct|union).*\.html",
        ),
        "driver": (
            "https://docs.nvidia.com/cuda/cuda-driver-api/",
            "modules.html",
            r"group__CUDA__.*\.html",
            "annotated.html",
            r"structCU.*\.html",
        ),
        "math": (
            "https://docs.nvidia.com/cuda/cuda-math-api/",
            "index.html",
            r"cuda_math_api/group__CUDA__MATH__.*\.html",
            "cuda_math_api/structs.html",
            r"struct.*\.html",
        ),
    }

    def __init__(
        self,
        api_type: str,
        output_dir: Path,
        skip_download: bool = False,
        force: bool = False,
    ):
        self.api_type = api_type
        base_url = self._CONFIG[api_type][0]
        super().__init__(
            base_url,
            output_dir,
            skip_download=skip_download,
            force=force,
        )

    def discover_pages(self) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Discover module and data structure pages."""
        if self.skip_download:
            modules_dir = self.cache_dir / "modules"
            structures_dir = self.cache_dir / "data-structures"
            modules = (
                [{"filename": f.stem} for f in sorted(modules_dir.glob("*.md"))]
                if modules_dir.exists()
                else []
            )
            structures = (
                [{"filename": f.stem} for f in sorted(structures_dir.glob("*.md"))]
                if structures_dir.exists()
                else []
            )
            return modules, structures

        modules = self._discover_modules()
        structures = self._discover_structures()
        return modules, structures

    def _discover_modules(self) -> list[dict[str, str]]:
        """Discover all module pages."""
        _, modules_path, modules_pattern, _, _ = self._CONFIG[self.api_type]
        page_url = urljoin(self.base_url, modules_path)
        soup = self.fetch_page(page_url)
        if not soup:
            return []

        modules = []
        seen = set()

        for link in soup.find_all("a", href=re.compile(modules_pattern)):
            href = link.get("href")
            title = link.get_text(strip=True)
            # Skip anchored links (e.g. page.html#func) — we want page-level granularity
            if href and title and "#" not in href and href not in seen:
                seen.add(href)
                modules.append(
                    {
                        "url": urljoin(page_url, href),
                        "filename": Path(href).name,
                        "title": title,
                    }
                )

        print(f"Discovered {len(modules)} module pages")
        return modules

    def _discover_structures(self) -> list[dict[str, str]]:
        """Discover all data structure pages."""
        _, _, _, structs_path, structs_pattern = self._CONFIG[self.api_type]
        page_url = urljoin(self.base_url, structs_path)
        try:
            soup = self.fetch_page(page_url)
        except Exception as e:
            print(f"Warning: Could not fetch {structs_path}: {e}")
            return []

        if not soup:
            return []

        structures = []
        seen = set()

        for link in soup.find_all("a", href=re.compile(structs_pattern)):
            href = link.get("href")
            title = link.get_text(strip=True)
            # Skip anchored links — we want page-level granularity
            if href and title and "#" not in href and href not in seen:
                seen.add(href)
                structures.append(
                    {
                        "url": urljoin(page_url, href),
                        "filename": Path(href).name,
                        "title": title,
                    }
                )

        print(f"Discovered {len(structures)} data structure pages")
        return structures

    def scrape_page(self, page_info: dict[str, str], output_path: Path) -> bool:
        """Scrape and save a single page."""
        if output_path.exists() and not self.force:
            print(f"  ✓ Using cached: {output_path.name}")
            return True

        try:
            soup = self.fetch_page(page_info["url"])
            if not soup:
                return False

            markdown = self.convert_to_markdown(soup, page_info["url"])
            header = f"# {page_info['title']}\n\n"
            header += f"**Source:** [{page_info['filename']}]({page_info['url']})\n\n"
            header += "---\n\n"

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(header + markdown, encoding="utf-8")

            print(f"  ✓ Saved: {output_path.name} ({len(header + markdown)} bytes)")
            return True
        except Exception as e:
            print(f"  ✗ Error scraping {page_info['url']}: {e}")
            return False

    def clean_markdown_file(self, file_path: Path) -> tuple[str, int, int]:
        """Clean a markdown file, returning (content, original_size, new_size)."""
        content = file_path.read_text(encoding="utf-8")
        original_size = len(content)

        # Remove duplicate function TOC
        content = self._remove_toc(content)

        # Remove duplicate headers
        content = re.sub(r"(### Functions\s*\n){2,}", "### Functions\n\n", content)

        # Remove footer
        footer_markers = [
            "![](https://docs.nvidia.com/cuda/common/formatting/NVIDIA-LogoBlack.svg)",
            "[Privacy Policy]",
            "Copyright ©",
        ]
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if any(marker in line for marker in footer_markers):
                content = "\n".join(lines[:i])
                break

        # Remove formatting artifacts
        content = content.replace("\n---\n", "\n")
        content = content.replace("\u200b", "")  # Zero-width spaces
        content = re.sub(r" \[inherited\]", "", content)

        # Remove anchor links
        content = re.sub(r"\[([^\]]+)\]\(#[^\)]+\)", r"\1", content)

        # Remove "See also:" sections
        content = self._remove_see_also(content)

        # Remove boilerplate notes
        boilerplate = [
            "Note that this function may also return error codes from previous, asynchronous launches.\n\n",
            "Note that this function may also return error codes from previous, asynchronous launches.",
        ]
        for text in boilerplate:
            content = content.replace(text, "")

        # Remove URLs from links (keep text only)
        content = re.sub(r"\[([^\]]+)\]\(https://[^)]+\)", r"\1", content)
        content = re.sub(r"\[\]\(https://[^)]+\)", "", content)

        # Clean up empty notes and trailing commas
        content = re.sub(r"\nNote:\n\n", "\n", content)
        content = re.sub(r",(\s*)$", r"\1", content, flags=re.MULTILINE)

        # Clean up whitespace
        content = re.sub(r"\n{4,}", "\n\n\n", content)
        content = "\n".join(line.rstrip() for line in content.split("\n"))

        return content, original_size, len(content)

    def _remove_toc(self, content: str) -> str:
        """Remove duplicate function TOC from content."""
        lines = content.split("\n")
        cleaned_lines = []
        in_toc = False
        seen_functions_header = False

        for line in lines:
            # Detect TOC lines (Driver API pattern)
            if (
                ") [" in line
                and "](#" in line
                and any(x in line for x in ["](https://", "CUresult", "CUdeviceptr"])
            ):
                in_toc = True
                continue

            # End of TOC
            if line.strip() == "### Functions":
                if seen_functions_header:
                    in_toc = False
                else:
                    seen_functions_header = True

            if not in_toc:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _remove_see_also(self, content: str) -> str:
        """Remove 'See also:' sections."""
        lines = content.split("\n")
        cleaned_lines = []
        in_see_also = False

        for line in lines:
            if line.strip() == "**See also:**":
                in_see_also = True
                continue

            if in_see_also:
                if (
                    line.startswith("#")
                    or line.startswith("[CUresult]")
                    or line.startswith("[void]")
                ):
                    in_see_also = False
                    cleaned_lines.append(line)
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def run(self) -> None:
        """Execute the scraping workflow."""
        print("=" * 70)
        print(f"CUDA {self.api_type.title()} API Documentation Scraper")
        print("=" * 70)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.skip_download:
            print("\n⚡ SKIP DOWNLOAD MODE - Using cached files")
        else:
            print("\n1. Discovering pages...")

        modules, structures = self.discover_pages()

        if not self.skip_download:
            print(
                f"\nTotal pages: {len(modules) + len(structures)} "
                f"(modules: {len(modules)}, structures: {len(structures)})"
            )

            modules_dir = self.cache_dir / "modules"
            structures_dir = self.cache_dir / "data-structures"
            modules_dir.mkdir(exist_ok=True)
            structures_dir.mkdir(exist_ok=True)

            # Scrape modules
            print("\n2. Scraping module pages...")
            for i, module in enumerate(modules, 1):
                print(f"\n[{i}/{len(modules)}] {module['title']}")
                filename = self.sanitize_filename(module["filename"]) + ".md"
                self.scrape_page(module, modules_dir / filename)

            # Scrape structures
            print("\n3. Scraping data structure pages...")
            for i, struct in enumerate(structures, 1):
                print(f"\n[{i}/{len(structures)}] {struct['title']}")
                filename = self.sanitize_filename(struct["filename"]) + ".md"
                self.scrape_page(struct, structures_dir / filename)

        # Cleanup phase
        print(f"\n{'4' if not self.skip_download else '1'}. Cleaning cached files...")
        cache_modules_dir = self.cache_dir / "modules"
        cache_structures_dir = self.cache_dir / "data-structures"
        out_modules_dir = self.output_dir / "modules"
        out_structures_dir = self.output_dir / "data-structures"
        out_modules_dir.mkdir(exist_ok=True)
        out_structures_dir.mkdir(exist_ok=True)

        total_original = 0
        total_new = 0
        files_cleaned = 0

        for md_file in sorted(cache_modules_dir.glob("*.md")):
            content, orig_size, new_size = self.clean_markdown_file(md_file)
            (out_modules_dir / md_file.name).write_text(content, encoding="utf-8")
            total_original += orig_size
            total_new += new_size
            files_cleaned += 1

        for md_file in sorted(cache_structures_dir.glob("*.md")):
            content, orig_size, new_size = self.clean_markdown_file(md_file)
            (out_structures_dir / md_file.name).write_text(content, encoding="utf-8")
            total_original += orig_size
            total_new += new_size
            files_cleaned += 1

        reduction = (
            (total_original - total_new) / total_original * 100
            if total_original > 0
            else 0
        )
        print(
            f"  Cleaned {files_cleaned} files: "
            f"{total_original:,} → {total_new:,} bytes ({reduction:.1f}% reduction)"
        )

        # Create index
        print(f"\n{'5' if not self.skip_download else '2'}. Creating index...")
        self._create_index(out_modules_dir, out_structures_dir)

        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)
        print(f"Output: {self.output_dir} ({total_new/1024/1024:.1f} MB)")

    def _create_index(self, modules_dir: Path, structures_dir: Path) -> None:
        """Create INDEX.md file."""
        modules = sorted(
            [
                {"title": f.stem.replace("-", " ").title(), "filename": f.stem}
                for f in modules_dir.glob("*.md")
            ],
            key=lambda x: x["title"],
        )
        structures = sorted(
            [
                {"title": f.stem.replace("-", " ").title(), "filename": f.stem}
                for f in structures_dir.glob("*.md")
            ],
            key=lambda x: x["title"],
        )

        content = f"# CUDA {self.api_type.title()} API Documentation Index\n\n"
        content += f"**Modules:** {len(modules)}  \n"
        content += f"**Data structures:** {len(structures)}  \n\n"

        content += "## Modules\n\n"
        for module in modules:
            filename = self.sanitize_filename(module["filename"]) + ".md"
            content += f"- [{module['title']}](modules/{filename})\n"

        content += "\n## Data Structures\n\n"
        for struct in structures:
            filename = self.sanitize_filename(struct["filename"]) + ".md"
            content += f"- [{struct['title']}](data-structures/{filename})\n"

        index_path = self.output_dir / "INDEX.md"
        index_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Created: {index_path}")


class SphinxScraper(DocumentationScraper):
    """Scraper for Sphinx single-page NVIDIA documentation.

    Handles any monolithic Sphinx doc page (cuBLAS, CUDA Math API, NVRTC, etc.)
    by splitting it into per-section markdown files organized into chapter dirs.
    """

    # Registry of known Sphinx docs: doc_type -> (display_name, page_url)
    KNOWN_DOCS: dict[str, tuple[str, str]] = {
        "cublas": (
            "cuBLAS",
            "https://docs.nvidia.com/cuda/cublas/index.html",
        ),
    }

    def __init__(self, display_name: str, page_url: str, output_dir: Path):
        self.display_name = display_name
        self.page_url = page_url
        # base_url = parent directory of page_url (used for absolute image URLs)
        base_url = page_url.rsplit("/", 1)[0] + "/"
        super().__init__(base_url, output_dir)

    @classmethod
    def from_doc_type(cls, doc_type: str, output_dir: Path) -> "SphinxScraper":
        """Construct from a registered doc_type key."""
        if doc_type not in cls.KNOWN_DOCS:
            raise ValueError(
                f"Unknown Sphinx doc type '{doc_type}'. "
                f"Known: {list(cls.KNOWN_DOCS)}"
            )
        display_name, page_url = cls.KNOWN_DOCS[doc_type]
        return cls(display_name, page_url, output_dir)

    def run(self) -> None:
        """Execute Sphinx single-page scraping workflow."""
        print("=" * 70)
        print(f"{self.display_name} Documentation Scraper")
        print("=" * 70)

        soup = self.fetch_page(self.page_url)
        if not soup:
            print("Failed to fetch documentation")
            return

        print("\nExtracting sections...")
        sections = self._extract_sections(soup)
        print(f"Found {len(sections)} sections")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Organize by top-level (h1) chapters
        current_chapter_dir = self.output_dir
        for section in sections:
            title_lower = section["title"].lower()
            if section["level"] == 0 and any(
                skip in title_lower for skip in ("notice", "acknowledgement")
            ):
                continue

            if section["level"] == 0:
                chapter_name = self.sanitize_filename(
                    section["title"], section["section_num"]
                )
                current_chapter_dir = self.output_dir / chapter_name
                current_chapter_dir.mkdir(parents=True, exist_ok=True)
                print(f"\nChapter: {section['title']}")

            self._save_section(section, current_chapter_dir)

        print(f"\n✓ Complete! Documentation saved to: {self.output_dir}")

    def _extract_sections(self, soup: BeautifulSoup) -> list[dict]:
        """Extract all h1-h4 sections from single-page Sphinx documentation."""
        content = None
        for selector in [
            {"role": "main"},
            {"class": "document"},
            {"class": "body"},
            {"itemprop": "articleBody"},
        ]:
            content = soup.find("div", selector) or soup.find("section", selector)
            if content:
                break

        if not content:
            return []

        sections = []
        headings = content.find_all(["h1", "h2", "h3", "h4"])

        for heading in headings:
            heading_text = heading.get_text(strip=True)
            if not heading_text:
                continue

            # Extract optional numeric section prefix (e.g. "2.3.1. Title")
            section_match = re.match(r"^(\d+(?:\.\d+)*)\.\s*(.+)$", heading_text)
            section_num = section_match.group(1) if section_match else ""
            title = section_match.group(2) if section_match else heading_text

            anchor_id = heading.get("id", "") or (
                heading.find("a").get("id", "") if heading.find("a") else ""
            )
            level = int(heading.name[1]) - 1  # h1→0, h2→1, h3→2, h4→3

            # Collect sibling elements until a heading of the same or higher level
            content_elements = []
            current = heading.next_sibling
            while current:
                if isinstance(current, Tag) and current.name in (
                    "h1", "h2", "h3", "h4"
                ):
                    if int(current.name[1]) - 1 <= level:
                        break
                if isinstance(current, Tag):
                    content_elements.append(current)
                current = current.next_sibling

            sections.append(
                {
                    "title": title,
                    "section_num": section_num,
                    "level": level,
                    "anchor": anchor_id,
                    "content": content_elements,
                }
            )

        return sections

    def _save_section(self, section: dict, parent_dir: Path) -> None:
        """Render a section as a markdown file and write it to parent_dir."""
        filename = self.sanitize_filename(section["title"], section["section_num"])
        markdown_parts = []

        # Heading
        level_prefix = "#" * (section["level"] + 1)
        title_with_num = (
            f"{section['section_num']}. {section['title']}"
            if section["section_num"]
            else section["title"]
        )
        markdown_parts.append(f"{level_prefix} {title_with_num}\n")

        # Body
        for element in section["content"]:
            for class_name in ("headerlink", "viewcode-link", "navigation", "related"):
                for unwanted in element.find_all(class_=class_name):
                    unwanted.decompose()

            md = self.h2t.handle(str(element))
            # Rewrite relative _images/ paths to absolute URLs
            md = re.sub(
                r"!\[(.*?)\]\(_images/(.*?)\)",
                rf"![\1]({self.base_url}_images/\2)",
                md,
            )
            if md:
                markdown_parts.append(md)

        markdown = "\n\n".join(markdown_parts)
        markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)

        output_file = parent_dir / f"{filename}.md"
        output_file.write_text(markdown, encoding="utf-8")
        print(f"  Saved: {output_file.name}")


class SphinxMultiPageScraper(DocumentationScraper):
    """Scraper for Sphinx multi-page NVIDIA documentation.

    Handles documentation sites with an index page linking to individual
    content pages (e.g., NCCL User Guide). Preserves URL directory structure
    in the output (usage/communicators.html → usage/communicators.md).
    """

    # Registry: doc_type -> (display_name, base_url, index_page)
    KNOWN_DOCS: dict[str, tuple[str, str, str]] = {
        "nccl": (
            "NCCL",
            "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/",
            "index.html",
        ),
    }

    # Non-content pages to skip
    SKIP_PAGES = {"genindex.html", "py-modindex.html", "search.html"}

    def __init__(
        self,
        display_name: str,
        base_url: str,
        index_page: str,
        output_dir: Path,
        force: bool = False,
    ):
        self.display_name = display_name
        self.index_page = index_page
        super().__init__(base_url, output_dir, force=force)

    @classmethod
    def from_doc_type(
        cls, doc_type: str, output_dir: Path, force: bool = False
    ) -> "SphinxMultiPageScraper":
        if doc_type not in cls.KNOWN_DOCS:
            raise ValueError(
                f"Unknown doc type '{doc_type}'. Known: {list(cls.KNOWN_DOCS)}"
            )
        display_name, base_url, index_page = cls.KNOWN_DOCS[doc_type]
        return cls(display_name, base_url, index_page, output_dir, force)

    def _discover_pages(self) -> list[dict[str, str]]:
        """Discover all content pages from the index TOC."""
        index_url = urljoin(self.base_url, self.index_page)
        soup = self.fetch_page(index_url)
        if not soup:
            return []

        pages = []
        seen: set[str] = set()

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            # Only clean page links — skip anchors, external, non-HTML
            if (
                "#" in href
                or href.startswith(("http://", "https://", "mailto:"))
                or not href.endswith(".html")
            ):
                continue
            basename = Path(href).name
            if basename in self.SKIP_PAGES or href in seen:
                continue
            seen.add(href)
            toc_title = link.get_text(strip=True)
            pages.append(
                {
                    "href": href,
                    "url": urljoin(index_url, href),
                    "toc_title": toc_title,
                }
            )

        return pages

    def run(self) -> None:
        """Execute Sphinx multi-page scraping workflow."""
        print("=" * 70)
        print(f"{self.display_name} Documentation Scraper")
        print("=" * 70)

        print("\n1. Discovering pages...")
        pages = self._discover_pages()
        print(f"  Found {len(pages)} content pages")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n2. Scraping pages...")
        all_pages: list[dict] = []
        for i, page in enumerate(pages, 1):
            output_path = self.output_dir / Path(page["href"]).with_suffix(".md")

            if output_path.exists() and not self.force:
                print(f"  [{i}/{len(pages)}] ✓ Cached: {page['href']}")
                # Read actual title from saved file
                first_line = output_path.read_text(encoding="utf-8").split("\n")[0]
                title = first_line.lstrip("# ").strip() or page["toc_title"]
                all_pages.append({"href": page["href"], "title": title})
                continue

            soup = self.fetch_page(page["url"])
            if not soup:
                continue

            # Use the page's own H1 as title (most accurate)
            h1 = soup.find("h1")
            title = (
                re.sub(r"^\d+(\.\d+)*\.?\s*", "", h1.get_text(strip=True))
                if h1
                else page["toc_title"]
            )

            markdown = self.convert_to_markdown(soup, page["url"])
            header = f"# {title}\n\n"
            header += f"**Source:** {page['url']}\n\n"
            header += "---\n\n"
            content = header + markdown

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
            print(
                f"  [{i}/{len(pages)}] ✓ {page['href']} ({len(content):,} bytes)"
            )
            all_pages.append({"href": page["href"], "title": title})

        print("\n3. Creating index...")
        self._create_index(all_pages)

        all_files = list(self.output_dir.rglob("*.md"))
        total_size = sum(f.stat().st_size for f in all_files)

        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)
        print(
            f"Output: {self.output_dir} "
            f"({len(all_files)} files, {total_size/1024/1024:.1f} MB)"
        )

    def _create_index(self, pages: list[dict]) -> None:
        """Create INDEX.md listing all scraped pages."""
        content = f"# {self.display_name} Documentation Index\n\n"
        content += f"**Pages:** {len(pages)}\n\n"
        for page in pages:
            md_rel = str(Path(page["href"]).with_suffix(".md"))
            content += f"- [{page['title']}]({md_rel})\n"
        index_path = self.output_dir / "INDEX.md"
        index_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Created: {index_path}")


class PTXScraper(SphinxScraper):
    """Scraper for PTX ISA single-page documentation."""

    def __init__(self, output_dir: Path):
        super().__init__(
            "PTX ISA",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html",
            output_dir,
        )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape CUDA documentation to markdown"
    )
    parser.add_argument(
        "api_type",
        choices=["ptx", "runtime", "driver", "math", "cublas", "nccl"],
        help="API type to scrape",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: cuda_skill/references/<api>-docs)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use cached files (runtime/driver only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cache exists",
    )

    args = parser.parse_args()

    # Set default output directory
    if not args.output_dir:
        default_dirs = {
            "ptx": "cuda_skill/references/ptx-docs",
            "runtime": "cuda_skill/references/cuda-runtime-docs",
            "driver": "cuda_skill/references/cuda-driver-docs",
            "math": "cuda_skill/references/cuda-math-docs",
            "cublas": "cuda_skill/references/cublas-docs",
            "nccl": "cuda_skill/references/nccl-docs",
        }
        args.output_dir = Path(default_dirs[args.api_type])

    # Create appropriate scraper
    scraper: PTXScraper | APIScraper | SphinxScraper | SphinxMultiPageScraper
    if args.api_type == "ptx":
        scraper = PTXScraper(args.output_dir)
    elif args.api_type == "cublas":
        scraper = SphinxScraper.from_doc_type("cublas", args.output_dir)
    elif args.api_type in SphinxMultiPageScraper.KNOWN_DOCS:
        scraper = SphinxMultiPageScraper.from_doc_type(
            args.api_type, args.output_dir, args.force
        )
    else:
        scraper = APIScraper(
            args.api_type, args.output_dir, args.skip_download, args.force
        )

    scraper.run()


if __name__ == "__main__":
    main()
