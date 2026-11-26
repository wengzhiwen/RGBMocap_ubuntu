# Converted FAH MANO import: validate the pickle stream, never load it.

import hashlib
import os
import pickletools
import uuid
from pathlib import Path

MAX_BYTES = 16 * 1024 * 1024 + 1
# Upstream FAH builds a MANO layer for each hand (it fits the right-hand model
# to a mirrored left crop, and its layers, fitter and Procrustes assert both
# files exist), so the left hand is registered the same way as the right.
HAND_FILES = {'right': 'MANO_RIGHT.pkl', 'left': 'MANO_LEFT.pkl'}
REQUIRED_FIELDS = (
    'v_template', 'shapedirs', 'posedirs', 'weights', 'f',
    'kintree_table', 'J_regressor', 'hands_mean', 'hands_components',
)

def _chumpy_message(name):
    return (
        f'{name} looks like a raw Chumpy or official MANO pickle. '
        'Convert the official MANO file with the upstream MANO conversion script, '
        f'then import the plain NumPy {name}.'
    )

def import_fah_mano_right(source):
    return import_fah_mano(source, 'right')

def import_fah_mano_left(source):
    return import_fah_mano(source, 'left')

def import_fah_mano(source, side='right'):
    """Register one converted MANO hand for FAH; an existing different file is never replaced."""
    from . import fah_runtime

    if side not in HAND_FILES:
        raise ValueError('Choose the MANO left or right hand.')
    name = HAND_FILES[side]

    if not fah_runtime.supported():
        raise ValueError('FAH MANO import is available on Linux only.')

    source = Path(source).resolve()
    if source.name != name:
        raise ValueError(f'Select a MANO file named {name}.')

    target = Path(fah_runtime.asset_roots()['mano']) / name

    try:
        with source.open('rb') as handle:
            data = handle.read(MAX_BYTES)
    except OSError as error:
        raise ValueError('Could not read the selected MANO file: ' + str(error)) from error
    if len(data) >= MAX_BYTES:
        raise ValueError(f'{name} is larger than 16 MiB; select the converted file.')

    _validate_mano_pickle(data, name)

    if target.exists():
        try:
            if _same_bytes(target, data):
                return target
            raise ValueError(f'A different {name} is already configured; it was not replaced.')
        except OSError as error:
            raise ValueError('Could not compare the existing MANO file: ' + str(error)) from error

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = _private_temp_file(target.parent, side)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temp_name, str(target))
        except FileExistsError:
            try:
                if _same_bytes(target, data):
                    return target
            except OSError:
                pass
            raise ValueError(f'A different {name} appeared; it was not replaced.')
        except OSError as error:
            raise ValueError(f'Could not publish {name}: ' + str(error)) from error
        finally:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    except OSError as error:
        raise ValueError(f'Could not publish {name}: ' + str(error)) from error

    return target

def _validate_mano_pickle(data, file_name='MANO_RIGHT.pkl'):
    text_opcode_names = {
        "STRING",
        "UNICODE",
        "BINSTRING",
        "SHORT_BINSTRING",
        "BINUNICODE",
        "SHORT_BINUNICODE",
        "BINUNICODE8",
        "GLOBAL",
    }

    required = set(REQUIRED_FIELDS)
    found = set()
    seen = False
    has_chumpy = False

    try:
        for opcode, arg, _ in pickletools.genops(data):
            seen = True
            opcode_name = opcode[0] if isinstance(opcode, tuple) else getattr(opcode, "name", str(opcode))
            if opcode_name not in text_opcode_names:
                continue

            values = arg if isinstance(arg, tuple) else (arg,)
            for value in values:
                if isinstance(value, bytes):
                    value = value.decode("latin1")
                if not isinstance(value, str):
                    continue
                if "chumpy" in value.lower():
                    has_chumpy = True
                    break
                if value in required:
                    found.add(value)

            if has_chumpy:
                break
    except Exception:
        if has_chumpy:
            raise ValueError(_chumpy_message(file_name))
        raise ValueError("Malformed pickle data.") from None

    if has_chumpy:
        raise ValueError(_chumpy_message(file_name))

    if not seen:
        raise ValueError("Empty pickle data.")

    missing = [field for field in REQUIRED_FIELDS if field not in found]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))


def _private_temp_file(directory, side='right'):
    temp = directory / ('.fah_mano_' + side + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return fd, str(temp)

def _same_bytes(path, data):
    if path.stat().st_size != len(data):
        return False
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest() == hashlib.sha256(data).hexdigest()
