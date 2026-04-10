"""
Dictionary manager module for Dugal Inventory System.

Rebuilt to serve three purposes:
  1. Build a semantic vocabulary map from the inventory document
     (full names, short triggers, family groupings, disambiguation)
  2. Feed that vocabulary into the search engine's learned_patterns
     so fuzzy matching is driven by real product names
  3. Expose a compact AI-context payload so the AI interpreter knows
     what items exist and how to disambiguate them before routing

The old phonetic-noise approach (random vowel swaps, dropped consonants)
is replaced by intent-driven variation generation:
  - brand/family triggers  -> map to the most common variant
  - specific sub-triggers  -> map to the exact variant
  - abbreviation aliases   -> double/dbl, reserve/res, etc.
  - spoken-English aliases -> user-defined ("green" -> Chartreuse Green)
"""

from __future__ import annotations

import os
import json
import re
import logging
import shutil
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Set, List, Optional, Any, Tuple, TYPE_CHECKING

from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QDialogButtonBox, QStatusBar, QProgressBar, QMessageBox, QStyle
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

ABBREV_MAP: Dict[str, List[str]] = {
    "double":   ["dbl", "dble"],
    "dbl":      ["double", "dble"],
    "reserve":  ["res", "rsrv", "resv"],
    "res":      ["reserve"],
    "single":   ["sngl", "sgl"],
    "barrel":   ["brl", "bar"],
    "straight": ["str8", "strt"],
    "whiskey":  ["whisky", "whsky"],
    "whisky":   ["whiskey", "whsky"],
    "proof":    ["prf"],
    "small":    ["sm", "sml"],
    "batch":    ["btch"],
    "edition":  ["ed", "edt"],
    "limited":  ["ltd", "lmtd"],
    "private":  ["prvt", "priv"],
    "select":   ["sel", "slct"],
    "original": ["orig"],
    "classic":  ["cls", "clsc"],
}

STOP_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "of", "with", "from", "by",
    "in", "on", "at", "to", "for", "de", "du", "le", "la", "les",
    "no", "nr", "vs", "etc", "co", "corp", "inc",
    "beer", "wine", "spirit", "spirits", "liquor", "alcohol",
}

MIN_TRIGGER_LEN = 3


# ---------------------------------------------------------------------------
# VocabularyMap
# ---------------------------------------------------------------------------

class VocabularyMap:
    """
    trigger (str, lowercase) -> canonical_name (str, original case)

    When multiple canonicals share the same trigger that trigger is
    stored in ambiguous_triggers instead of trigger_map.
    """

    def __init__(self):
        self.trigger_map: Dict[str, str] = {}
        self.ambiguous_triggers: Dict[str, Set[str]] = {}
        self.reverse_map: Dict[str, Set[str]] = defaultdict(set)
        self.families: Dict[str, List[str]] = defaultdict(list)
        self._sorted_pairs: Optional[List[Tuple[str, str]]] = None

    def add(self, trigger: str, canonical: str) -> None:
        trigger = trigger.strip().lower()
        if not trigger or len(trigger) < MIN_TRIGGER_LEN:
            return
        self._sorted_pairs = None
        if trigger in self.trigger_map:
            existing = self.trigger_map[trigger]
            if existing == canonical:
                return
            del self.trigger_map[trigger]
            self.ambiguous_triggers[trigger] = {existing, canonical}
        elif trigger in self.ambiguous_triggers:
            self.ambiguous_triggers[trigger].add(canonical)
        else:
            self.trigger_map[trigger] = canonical
        self.reverse_map[canonical].add(trigger)

    def resolve(self, text: str) -> Optional[str]:
        return self.trigger_map.get(text.strip().lower())

    def is_ambiguous(self, trigger: str) -> bool:
        return trigger.strip().lower() in self.ambiguous_triggers

    def candidates(self, trigger: str) -> List[str]:
        key = trigger.strip().lower()
        if key in self.trigger_map:
            return [self.trigger_map[key]]
        return sorted(self.ambiguous_triggers.get(key, set()))

    def sorted_pairs(self) -> List[Tuple[str, str]]:
        if self._sorted_pairs is None:
            self._sorted_pairs = sorted(
                self.trigger_map.items(), key=lambda x: len(x[0]), reverse=True
            )
        return self._sorted_pairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_map": self.trigger_map,
            "ambiguous_triggers": {k: list(v) for k, v in self.ambiguous_triggers.items()},
            "families": dict(self.families),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VocabularyMap":
        vm = cls()
        vm.trigger_map = data.get("trigger_map", {})
        vm.ambiguous_triggers = {
            k: set(v) for k, v in data.get("ambiguous_triggers", {}).items()
        }
        vm.families = defaultdict(list, data.get("families", {}))
        for trigger, canonical in vm.trigger_map.items():
            vm.reverse_map[canonical].add(trigger)
        return vm

    def summary(self) -> str:
        return (
            f"{len(self.trigger_map)} unambiguous triggers, "
            f"{len(self.ambiguous_triggers)} ambiguous, "
            f"{len(self.reverse_map)} canonical items"
        )


# ---------------------------------------------------------------------------
# VocabularyBuilder
# ---------------------------------------------------------------------------

class VocabularyBuilder:
    """
    Turns a list of canonical inventory names into a VocabularyMap.

    Per name:
      1. Full-name trigger
      2. Abbreviation variants (ABBREV_MAP)
      3. Family trigger (first meaningful word) -- safe default if unique
      4. Prefix triggers (first 2..N-1 words)
      5. Specific disambiguators for multi-member families
    """

    def __init__(self):
        self.vm = VocabularyMap()

    def build(self, canonical_names: List[str]) -> VocabularyMap:
        families: Dict[str, List[str]] = defaultdict(list)
        for name in canonical_names:
            words = self._words(name)
            if words:
                families[words[0]].append(name)
        for family_key, members in families.items():
            self.vm.families[family_key] = members
        for name in canonical_names:
            self._process(name, families)
        logger.debug(f"VocabularyBuilder: {self.vm.summary()}")
        return self.vm

    def _process(self, name: str, families: Dict[str, List[str]]) -> None:
        words = self._words(name)
        if not words:
            return
        # 1. full name
        self.vm.add(name.lower(), name)
        # 2. abbreviation variants
        for variant in self._abbrev_variants(words, name):
            self.vm.add(variant, name)
        # 3. family trigger
        family_key = words[0]
        if len(families.get(family_key, [])) == 1:
            self.vm.add(family_key, name)
        # 4. prefix triggers
        for i in range(2, len(words)):
            self.vm.add(" ".join(words[:i]), name)
        # 5. specific disambiguators for multi-member families
        if len(families.get(family_key, [])) > 1:
            for i in range(2, len(words) + 1):
                self.vm.add(" ".join(words[:i]), name)

    def _words(self, name: str) -> List[str]:
        text = unicodedata.normalize("NFKD", name.lower())
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [w for w in text.split() if w and w not in STOP_WORDS]

    def _abbrev_variants(self, words: List[str], canonical: str) -> List[str]:
        variants = []
        for i, word in enumerate(words):
            if word in ABBREV_MAP:
                for abbrev in ABBREV_MAP[word]:
                    new_words = words[:i] + [abbrev] + words[i + 1:]
                    variants.append(" ".join(new_words))
        return variants

    def add_alias(self, spoken_phrase: str, canonical: str) -> None:
        self.vm.add(spoken_phrase, canonical)


# ---------------------------------------------------------------------------
# DictionaryManager
# ---------------------------------------------------------------------------

class DictionaryManager(QWidget):
    """
    Qt widget + integration glue.

    Key public methods
    ------------------
    process_inventory_terms(workbook, search_engine)
        Build VocabularyMap from inventory, push to search engine.

    get_ai_context_payload()
        Return compact dict for voice_interaction._build_ai_context().

    add_alias(spoken_phrase, canonical)
        Add a custom spoken alias.
    """

    term_added    = pyqtSignal(dict)
    term_updated  = pyqtSignal(dict)
    scan_completed = pyqtSignal(dict)
    pattern_learned = pyqtSignal(str, str)

    VOCAB_CACHE_FILE = os.path.join(".dugal_data", "vocabulary_map.json")

    def __init__(self, search_engine=None, dugal=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing Dictionary Manager")

        self.search_engine = search_engine
        self.dugal = dugal
        self.logging_manager = getattr(dugal, "logging_manager", None) if dugal else None

        self.vocab_map: VocabularyMap = VocabularyMap()
        self.custom_aliases: Dict[str, str] = {}

        self.dictionary_tree = QTreeWidget()
        self.dictionary_tree.setHeaderLabels(["Trigger / Spoken Phrase", "Resolves To"])
        self.dictionary_tree.setColumnCount(2)

        self.scan_button  = None
        self.add_button   = None
        self.search_bar   = None
        self.progress_bar = None
        self.pattern_editor = None

        try:
            self.setup_icons()
        except Exception as e:
            self.logger.error(f"Icon setup error: {e}")
            self.base_term_icon = QIcon()
            self.variation_icon = QIcon()

        self.pattern_storage = getattr(dugal, "data_manager", None) if dugal else None
        self.temp_dir = os.path.join(os.getcwd(), ".dugal_temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.logger.debug("Setting up Dictionary Manager UI")
        try:
            self.setup_ui()
        except Exception as e:
            self.logger.error(f"UI setup error: {e}")
            raise

        try:
            self._load_vocab_cache()
            self.refresh_dictionary_view()
        except Exception as e:
            self.logger.error(f"Cache load error: {e}")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def setup_icons(self):
        style = self.style()
        self.base_term_icon = QIcon(style.standardPixmap(QStyle.SP_DirIcon))
        self.variation_icon = QIcon(style.standardPixmap(QStyle.SP_FileIcon))
        self.logger.debug("Icons setup completed successfully")

    def setup_ui(self):
        logger.debug("Setting up Dictionary Manager UI")
        layout = QVBoxLayout(self)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        button_panel = QHBoxLayout()
        self.scan_button = QPushButton("Scan Document for Terms")
        self.scan_button.setStyleSheet(
            "QPushButton{background:#6c5ce7;color:white;padding:12px;"
            "border-radius:5px;font-weight:bold;min-width:200px}"
            "QPushButton:hover{background:#5f51e8}"
        )
        self.add_button = QPushButton("Add Spoken Alias")
        self.add_button.setStyleSheet(
            "QPushButton{background:#00b894;color:white;padding:12px;"
            "border-radius:5px;font-weight:bold;min-width:200px}"
            "QPushButton:hover{background:#00a885}"
        )
        button_panel.addWidget(self.scan_button)
        button_panel.addWidget(self.add_button)
        button_panel.addStretch()

        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search triggers or product names...")
        self.search_bar.setStyleSheet(
            "QLineEdit{padding:8px;border:2px solid #ddd;border-radius:5px;font-size:14px}"
        )
        search_layout.addWidget(self.search_bar)

        self.dictionary_tree = QTreeWidget()
        self.dictionary_tree.setHeaderLabels(["Trigger / Spoken Phrase", "Resolves To"])
        self.dictionary_tree.setStyleSheet(
            "QTreeWidget{border:2px solid #ddd;border-radius:5px;padding:5px}"
            "QTreeWidget::item{padding:5px}"
        )
        self.dictionary_tree.setColumnWidth(0, 280)

        layout.addLayout(button_panel)
        layout.addLayout(search_layout)
        layout.addWidget(self.dictionary_tree)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("QStatusBar{border-top:1px solid #ddd;padding:5px}")
        layout.addWidget(self.status_bar)

        self.scan_button.clicked.connect(self.scan_document)
        self.add_button.clicked.connect(self.show_add_alias_dialog)
        self.search_bar.textChanged.connect(self.filter_terms)
        self.dictionary_tree.itemDoubleClicked.connect(self.edit_term)

        logger.debug("Dictionary Manager UI setup complete")
        self.refresh_dictionary_view()

    def show_status_message(self, message: str, error: bool = False) -> None:
        try:
            self.status_bar.setStyleSheet("color:red" if error else "")
            self.status_bar.showMessage(message, 3000)
        except Exception as e:
            self.logger.error(f"Status message error: {e}")

    # ------------------------------------------------------------------
    # Core vocabulary building
    # ------------------------------------------------------------------

    def process_inventory_terms(self, workbook=None, search_engine=None) -> int:
        """
        Build VocabularyMap from inventory cache.
        Push trigger->canonical pairs into search_engine.learned_patterns.
        Returns number of entries pushed.
        """
        logger.debug("Processing inventory terms for vocabulary map")

        if search_engine:
            self.search_engine = search_engine
        se = self.search_engine
        if not se or not hasattr(se, "inventory_cache"):
            logger.warning("No search engine with inventory cache -- skipping")
            return 0

        # Collect canonical names
        canonical_names: List[str] = []
        seen: Set[str] = set()
        for entries in se.inventory_cache.values():
            for entry in entries:
                original = entry.get("original", "")
                if original and original not in seen:
                    seen.add(original)
                    canonical_names.append(original)

        if not canonical_names:
            logger.warning("Inventory cache empty -- no vocabulary to build")
            return 0

        logger.debug(f"Building vocabulary from {len(canonical_names)} canonical names")

        builder = VocabularyBuilder()
        self.vocab_map = builder.build(canonical_names)

        # Re-inject custom aliases
        for spoken, canonical in self.custom_aliases.items():
            self.vocab_map.add(spoken, canonical)

        variations_added = self._push_to_search_engine(se)
        self._save_vocab_cache()

        logger.debug(
            f"Vocabulary build complete: {self.vocab_map.summary()}, "
            f"{variations_added} entries pushed to search engine"
        )
        self.refresh_dictionary_view()
        return variations_added

    def process_inventory_terms_async(self, workbook=None, search_engine=None):
        with ThreadPoolExecutor() as executor:
            return executor.submit(self.process_inventory_terms, workbook, search_engine)

    # ------------------------------------------------------------------
    # Search engine integration
    # ------------------------------------------------------------------

    def _push_to_search_engine(self, search_engine) -> int:
        """Write trigger->canonical pairs into search_engine.learned_patterns."""
        if not hasattr(search_engine, "learned_patterns"):
            search_engine.learned_patterns = {}

        count = 0
        for trigger, canonical in self.vocab_map.trigger_map.items():
            canon_key = canonical.lower()
            if canon_key not in search_engine.learned_patterns:
                search_engine.learned_patterns[canon_key] = set()
            search_engine.learned_patterns[canon_key].add(trigger)
            count += 1

            # Per-word sub-triggers so "woodford" hits "Woodford Reserve"
            words = trigger.split()
            if len(words) > 1:
                for word in words:
                    if len(word) >= MIN_TRIGGER_LEN and word not in STOP_WORDS:
                        if word not in search_engine.learned_patterns:
                            search_engine.learned_patterns[word] = set()
                        search_engine.learned_patterns[word].add(trigger)
                        count += 1
        return count

    # ------------------------------------------------------------------
    # AI context payload
    # ------------------------------------------------------------------

    def get_ai_context_payload(self, max_items: int = 200) -> Dict[str, Any]:
        """
        Return a compact dict for _build_ai_context() to merge into the
        context passed to the AI interpreter.

        Keys:
          inventory_sample   : up to max_items canonical names
          family_map         : brand -> [variants]  (disambiguation)
          ambiguous_triggers : triggers needing clarification
          spoken_aliases     : custom phrase -> canonical
          vocab_size         : total canonical count
        """
        if not self.vocab_map.reverse_map:
            return {}

        all_canonicals = list(self.vocab_map.reverse_map.keys())
        sample = all_canonicals[:max_items]

        family_map = {
            family: members
            for family, members in self.vocab_map.families.items()
            if len(members) > 1
        }

        ambiguous = list(self.vocab_map.ambiguous_triggers.keys())[:50]

        return {
            "inventory_sample":   sample,
            "family_map":         family_map,
            "ambiguous_triggers": ambiguous,
            "spoken_aliases":     self.custom_aliases,
            "vocab_size":         len(all_canonicals),
        }

    # ------------------------------------------------------------------
    # Custom alias management
    # ------------------------------------------------------------------

    def add_alias(self, spoken_phrase: str, canonical: str) -> bool:
        try:
            spoken_phrase = spoken_phrase.strip().lower()
            if not spoken_phrase or not canonical:
                return False
            self.custom_aliases[spoken_phrase] = canonical
            self.vocab_map.add(spoken_phrase, canonical)
            if self.search_engine:
                self._push_to_search_engine(self.search_engine)
            self._save_vocab_cache()
            self.refresh_dictionary_view()
            self.show_status_message(f"Alias added: '{spoken_phrase}' -> '{canonical}'")
            self.pattern_learned.emit(spoken_phrase, canonical)
            return True
        except Exception as e:
            self.logger.error(f"Error adding alias: {e}")
            return False

    def show_add_alias_dialog(self) -> None:
        try:
            dialog = AddAliasDialog(parent=self)
            if dialog.exec_():
                spoken, canonical = dialog.get_values()
                if spoken and canonical:
                    self.add_alias(spoken, canonical)
        except Exception as e:
            self.logger.error(f"Error showing alias dialog: {e}")

    # ------------------------------------------------------------------
    # Search engine accessor
    # ------------------------------------------------------------------

    def get_search_engine(self):
        try:
            from component_manager import component_manager
            se = component_manager.get_search_engine()
            if se:
                self.search_engine = se
                return se
        except ImportError:
            pass
        try:
            from global_registry import GlobalRegistry
            se = GlobalRegistry.get("search_engine")
            if se:
                self.search_engine = se
                return se
        except Exception:
            pass
        return getattr(self, "search_engine", None)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_vocab_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.VOCAB_CACHE_FILE), exist_ok=True)
            data = self.vocab_map.to_dict()
            data["custom_aliases"] = self.custom_aliases
            with open(self.VOCAB_CACHE_FILE, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            logger.debug(f"Vocabulary cache saved")
        except Exception as e:
            logger.error(f"Error saving vocabulary cache: {e}")

    def _load_vocab_cache(self) -> None:
        try:
            if not os.path.exists(self.VOCAB_CACHE_FILE):
                return
            with open(self.VOCAB_CACHE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.vocab_map = VocabularyMap.from_dict(data)
            self.custom_aliases = data.get("custom_aliases", {})
            logger.debug(f"Vocabulary cache loaded: {self.vocab_map.summary()}")
        except Exception as e:
            logger.error(f"Error loading vocabulary cache: {e}")

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def load_patterns(self) -> None:
        try:
            if self.pattern_storage:
                patterns = self.pattern_storage.load_patterns()
                if patterns:
                    for base_term, variations in patterns.items():
                        for variation in (variations if isinstance(variations, (list, set)) else []):
                            self.vocab_map.add(variation, base_term)
            self.logger.debug(f"Loaded legacy patterns; vocab: {self.vocab_map.summary()}")
        except Exception as e:
            self.logger.error(f"Error loading legacy patterns: {e}")

    def add_pattern(self, base_term: str, variation: str) -> None:
        base_term = base_term.strip()
        variation = variation.strip().lower()
        if not base_term or not variation:
            return
        self.vocab_map.add(variation, base_term)
        if self.search_engine:
            self._push_to_search_engine(self.search_engine)
        if self.pattern_storage:
            self.pattern_storage.add_pattern(base_term, variation)
        self.pattern_learned.emit(base_term, variation)
        self.refresh_dictionary_view()

    def generate_phonetic_variations(self, term: str) -> List[str]:
        """Backward-compat stub -- returns vocabulary-map triggers instead of noise."""
        triggers = list(self.vocab_map.reverse_map.get(term, set()))
        words = re.sub(r"[^a-z0-9\s]", " ", term.lower()).split()
        for i, word in enumerate(words):
            if word in ABBREV_MAP:
                for abbrev in ABBREV_MAP[word]:
                    triggers.append(" ".join(words[:i] + [abbrev] + words[i + 1:]))
        return list(set(triggers))

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def refresh_dictionary_view(self) -> None:
        try:
            self.dictionary_tree.clear()
            canonical_to_triggers: Dict[str, List[str]] = defaultdict(list)
            for trigger, canonical in self.vocab_map.trigger_map.items():
                canonical_to_triggers[canonical].append(trigger)
            count = 0
            for canonical in sorted(canonical_to_triggers.keys()):
                triggers = sorted(canonical_to_triggers[canonical], key=len)
                base_item = QTreeWidgetItem(self.dictionary_tree)
                base_item.setText(0, triggers[0] if triggers else canonical)
                base_item.setText(1, canonical)
                base_item.setIcon(0, self.base_term_icon)
                for trigger in triggers[1:]:
                    child = QTreeWidgetItem(base_item)
                    child.setText(0, trigger)
                    child.setText(1, canonical)
                    child.setIcon(0, self.variation_icon)
                count += 1
            self.logger.debug(f"Loaded {count} canonical items into view")
        except Exception as e:
            self.logger.error(f"Error refreshing dictionary view: {e}")

    def filter_terms(self, search_text: str) -> None:
        try:
            search_text = search_text.lower().strip()
            for i in range(self.dictionary_tree.topLevelItemCount()):
                base_item = self.dictionary_tree.topLevelItem(i)
                t0 = base_item.text(0).lower()
                t1 = base_item.text(1).lower()
                matches = search_text in t0 or search_text in t1
                if not matches:
                    for j in range(base_item.childCount()):
                        if search_text in base_item.child(j).text(0).lower():
                            matches = True
                            break
                base_item.setHidden(not matches)
                if matches:
                    base_item.setExpanded(True)
        except Exception as e:
            self.logger.error(f"Error filtering terms: {e}")

    def find_or_create_base_term(self, base_term: str) -> QTreeWidgetItem:
        root = self.dictionary_tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.text(1).lower() == base_term.lower():
                return item
        new_item = QTreeWidgetItem(self.dictionary_tree)
        new_item.setText(0, base_term)
        new_item.setText(1, base_term)
        new_item.setIcon(0, self.base_term_icon)
        return new_item

    def edit_term(self, item: QTreeWidgetItem, column: int) -> None:
        try:
            spoken = item.text(0)
            canonical = item.text(1)
            dialog = AddAliasDialog(parent=self, spoken=spoken, canonical=canonical)
            if dialog.exec_():
                new_spoken, new_canonical = dialog.get_values()
                if new_spoken and new_canonical:
                    self.add_alias(new_spoken, new_canonical)
        except Exception as e:
            self.logger.error(f"Error editing term: {e}")

    def scan_document(self):
        logger.debug("Initiating document scan")
        se = self.get_search_engine()
        if not se or not getattr(se, "inventory_cache", None):
            QMessageBox.warning(self, "No Inventory", "Load a workbook first.")
            return
        reply = QMessageBox.question(
            self, "Scan Document",
            "Rebuild vocabulary map from current inventory?"
        )
        if reply == QMessageBox.Yes:
            self.scan_button.setEnabled(False)
            self.progress_bar.show()
            self.progress_bar.setValue(10)
            try:
                n = self.process_inventory_terms(search_engine=se)
                self.progress_bar.setValue(100)
                QMessageBox.information(
                    self, "Done",
                    f"Vocabulary rebuilt: {self.vocab_map.summary()}\n"
                    f"{n} entries pushed to search engine."
                )
                self.scan_completed.emit({"triggers": n})
            except Exception as e:
                logger.error(f"Scan error: {e}")
                QMessageBox.warning(self, "Error", str(e))
            finally:
                self.scan_button.setEnabled(True)
                self.progress_bar.hide()

    def save_current_state(self):
        self._save_vocab_cache()

    def cleanup(self):
        try:
            self.save_current_state()
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            if self.pattern_editor:
                self.pattern_editor.close()
            for widget in self.findChildren(QWidget):
                try:
                    widget.close()
                    widget.deleteLater()
                except Exception:
                    pass
            logger.debug("Dictionary manager cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def closeEvent(self, event):
        try:
            self.cleanup()
        except Exception:
            pass
        event.accept()


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------

class AddAliasDialog(QDialog):
    def __init__(self, parent=None, spoken: str = "", canonical: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Add Spoken Alias")
        self._build_ui(spoken, canonical)

    def _build_ui(self, spoken: str, canonical: str):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("What you say (spoken phrase):"))
        self._spoken = QLineEdit(spoken)
        self._spoken.setPlaceholderText('e.g. "green" or "larceny barrel"')
        layout.addWidget(self._spoken)
        layout.addWidget(QLabel("Resolves to (exact product name):"))
        self._canonical = QLineEdit(canonical)
        self._canonical.setPlaceholderText('e.g. "Chartreuse Green" or "Larceny Barrel Proof"')
        layout.addWidget(self._canonical)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> Tuple[str, str]:
        return self._spoken.text().strip(), self._canonical.text().strip()

    def accept(self):
        if not self._spoken.text().strip() or not self._canonical.text().strip():
            QMessageBox.warning(self, "Validation", "Both fields are required.")
            return
        super().accept()


# Backward-compat aliases
AddTermDialog = AddAliasDialog
TermEditorDialog = AddAliasDialog


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    dm = DictionaryManager(None)
    dm.show()
    sys.exit(app.exec_())
