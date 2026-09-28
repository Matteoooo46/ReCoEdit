import json
import tempfile
import unittest
from pathlib import Path

from build_training_annotations import build


class AnnotationExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Existence validation only; image decoding is deliberately a separate audit.
        (self.root/'ref.jpg').touch()
        (self.root/'target.jpg').touch()
        self.row = dict(sample_id='p1_f0', product_id='p1', source='fixture',
                        references=['ref.jpg'], target='target.jpg',
                        original_prompt='original', editor_prompt='editor',
                        caption='caption', rewriter_instruction='rewrite')

    def run_build(self, rows):
        path = self.root/'input.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return build(path, self.root, self.root/'out')

    def test_roles_and_distinct_text(self):
        self.run_build([self.row])
        editor = json.loads((self.root/'out/editor.jsonl').read_text())
        rewriter = json.loads((self.root/'out/rewriter.jsonl').read_text())
        self.assertEqual(editor['image'], 'target.jpg')
        self.assertEqual(editor['edit_image'], ['ref.jpg'])
        self.assertEqual(editor['prompt'], 'editor')
        self.assertEqual(rewriter['output'], 'caption')
        self.assertEqual(rewriter['input'], 'original')
        with self.assertRaises(FileExistsError):
            self.run_build([self.row])

    def test_bad_inputs_do_not_create_output(self):
        cases = [[self.row, self.row], [{**self.row, 'target': 'ref.jpg'}],
                 [{**self.row, 'caption': 'ERROR: failure'}],
                 [{**self.row, 'references': ['missing.jpg']}],
                 [{**self.row, 'target': '/tmp/target.jpg'}],
                 [{**self.row, 'editor_prompt': ''}]]
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.run_build(rows)
                self.assertFalse((self.root/'out').exists())


if __name__ == '__main__':
    unittest.main()
